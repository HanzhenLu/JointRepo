# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 
import numpy as np
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from typing import Tuple, List

from utils.util import Example, split_into_smaller_blocks, CodeBlock, bm25_retrieve, cross_file_contexts

logger = logging.getLogger(__name__)

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.sep_token:
        tokenizer.add_special_tokens(
            {'sep_token': '<SEP>'}
        )
        logger.info("Add <SEP> into the tokenizer")
    generator = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if hasattr(args, "weighted_parameters") and args.weighted_parameters is not None:
        logger.info(f"loadding parameters from {args.weighted_parameters}")
        generator.load_state_dict(torch.load(args.weighted_parameters))

    logger.info("Finish loading generator [%s] from %s", get_model_size(generator), args.model_name_or_path)

    return generator, tokenizer

class CustomDataset(Dataset):
    """A dataset class for code generation

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_codeblocks: Retrieved code blocks.
    """
    
    def __init__(self, args, tokenizer:PreTrainedTokenizer, examples:List[Example]) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.special_tokens = {
            "prefix_id": tokenizer.convert_tokens_to_ids("<PREFIX>"),
            "suffix_id": tokenizer.convert_tokens_to_ids("<SUFFIX>"),
            "middle_id": tokenizer.convert_tokens_to_ids("<MIDDLE>")
        }
        self.input_ids = []
        self.attention_mask = []
        for example in tqdm(examples):
            input_ids, attention_mask = self.construct_prompt(example).values()
            self.input_ids.append(torch.tensor(input_ids)) 
            self.attention_mask.append(torch.tensor(attention_mask))

    def __len__(self):
        return len(self.examples)
    
    def construct_prompt(self, example:Example):
        '''
        concatenate the context from other files with the context within the current file
        '''
        prefix = example.prefix
        suffix = example.suffix
            
        # TODO: set the cross_file_budget as a hyperparameter
        cross_file_budget = int(0.75 * self.args.max_input_length)
        repo_content = cross_file_contexts(example.relevant_code, self.tokenizer, cross_file_budget)
        
        prefix_tokenized_result = self.tokenizer(prefix, add_special_tokens=False)
        suffix_tokenized_result = self.tokenizer(suffix, add_special_tokens=False)
        
        left_budget = self.args.max_input_length - len(repo_content["input_ids"]) - 3
        prefix_length = int(left_budget / 2)
        suffix_length = int(left_budget - prefix_length)
        if len(prefix_tokenized_result["input_ids"]) < prefix_length and len(suffix_tokenized_result["input_ids"]) < suffix_length:
            prefix_ids = prefix_tokenized_result["input_ids"]
            suffix_ids = suffix_tokenized_result["input_ids"]
        elif len(prefix_tokenized_result["input_ids"]) < prefix_length:
            prefix_ids = prefix_tokenized_result["input_ids"]
            suffix_length = int(left_budget - len(prefix_ids))
            suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
        elif len(suffix_tokenized_result["input_ids"]) < suffix_length:
            suffix_ids = suffix_tokenized_result["input_ids"]
            prefix_length = int(left_budget - len(suffix_ids))
            prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
        else:
            prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
            suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
        
        input_ids = [self.special_tokens["suffix_id"]] + suffix_ids + [self.special_tokens["prefix_id"]] + repo_content["input_ids"] + prefix_ids + [self.special_tokens["middle_id"]]
        attention_mask = [1] * len(input_ids)
        padding_length = self.args.max_input_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        attention_mask = [0] * padding_length + attention_mask
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask
        }
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.attention_mask[index]

def generate(args, generator:PreTrainedModel, tokenizer:PreTrainedTokenizer, examples:List[Example]) -> List[str]:
    generated_codes = []
    dataset = CustomDataset(args, tokenizer, examples)
    sampler = SequentialSampler(dataset)
    # TODO: set the num_workers as a hyperparameter
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
    pbar = tqdm(dataloader, desc="Generating")
    with torch.no_grad():
        for batch in pbar:
            input_ids, attention_mask = [x.to(args.device) for x in batch]
            generated_ids = generator.generate(input_ids, attention_mask=attention_mask, 
                max_length=input_ids.size(1)+args.max_generation_length, pad_token_id=tokenizer.pad_token_id)
            generated_codes.extend(generated_ids[:, input_ids.size(1):])
    return [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]