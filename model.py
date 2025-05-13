# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 
import numpy as np
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from typing import Tuple, List, Union

from utils.util import Example, CodeBlock, cross_file_contexts

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
        if "llama" == args.model_type:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<PREFIX>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<SUFFIX>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<MIDDLE>"),
                "eos_id": self.tokenizer.convert_tokens_to_ids("<EOS>")
            }
        elif "deepseek" == args.model_type:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁end｜>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>"),
                "eos_id": self.tokenizer.eos_token_id
            }
        elif "Qwen" == args.model_type:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<|fim_prefix|>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<|fim_suffix|>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<|fim_middle|>"),
                "repo_name_id": self.tokenizer.convert_tokens_to_ids("<|repo_name|>"),
                "file_sep_id": self.tokenizer.convert_tokens_to_ids("<|file_sep|>")
            }
        else:
            raise RuntimeError(f"Unsupport model type {args.model_type}")
        self.input_ids = []
        self.attention_mask = []
        for example in tqdm(examples):
            if args.ntp:
                input_ids, attention_mask = self.construct_ntp_prompt(example).values()
            else:
                input_ids, attention_mask = self.construct_fim_prompt(example).values()
            self.input_ids.append(torch.tensor(input_ids)) 
            self.attention_mask.append(torch.tensor(attention_mask))

    def __len__(self):
        return len(self.examples)
    
    def construct_fim_prompt(self, example:Example):
        '''
        concatenate the context from other files with the context within the current file
        '''
        prefix = example.prefix
        suffix = example.suffix
            
        # TODO: set the cross_file_budget as a hyperparameter
        cross_file_budget = int(0.75 * self.args.max_input_length)
        repo_content = cross_file_contexts(example.relevant_code[:self.args.relevant_code_num], self.tokenizer, cross_file_budget)
        
        prefix_tokenized_result = self.tokenizer(prefix, add_special_tokens=False)
        suffix_tokenized_result = self.tokenizer(suffix, add_special_tokens=False)
        
        left_budget = self.args.max_input_length - len(repo_content["input_ids"]) - 4
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
        
        if "llama" == self.args.model_type:
            input_ids = [self.special_token_ids["suffix_id"]] + suffix_ids + [self.special_token_ids["prefix_id"]] \
                + repo_content["input_ids"] + prefix_ids + [self.special_token_ids["middle_id"]]
        elif "deepseek" == self.args.model_type:
            input_ids = [32013] + [self.special_token_ids["prefix_id"]] + repo_content["input_ids"] \
                + prefix_ids + [self.special_token_ids["middle_id"]] + suffix_ids + [self.special_token_ids["suffix_id"]]
        elif "Qwen" == self.args.model_type:
            input_ids = [self.special_token_ids["prefix_id"]] + repo_content["input_ids"] + \
                prefix_ids + [self.special_token_ids["suffix_id"]] + \
                suffix_ids + [self.special_token_ids["middle_id"]]
        else:
            raise RuntimeError(f"Unsupport model type {self.args.model_type}")
        attention_mask = [1] * len(input_ids)
        padding_length = self.args.max_input_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        attention_mask = [0] * padding_length + attention_mask
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask
        }
    
    def construct_ntp_prompt(self, example:Example):
        prefix = example.prefix
        cross_file_budget = int(0.25 * self.args.max_input_length)
        repo_content = cross_file_contexts(example.relevant_code[:self.args.relevant_code_num], self.tokenizer, cross_file_budget)
        prefix_tokenized_result = self.tokenizer(prefix, add_special_tokens=False)
        context_budget = int(0.75 * self.args.max_input_length)
        prefix_ids = prefix_tokenized_result["input_ids"][-context_budget:]
        if self.args.model_type == "llama":
            input_ids = repo_content["input_ids"] + prefix_ids
        elif self.args.model_type == "deepseek":
            input_ids = [32013] + repo_content["input_ids"] + prefix_ids
        elif self.args.model_type == "Qwen":
            input_ids = repo_content["input_ids"] + prefix_ids
        else:
            raise RuntimeError(f"Unsupport model type {self.args.model_type}")
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

class Retriever:
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if "snowflake" in model_name_or_path.lower():
            self.model = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=False, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def embedding(self, input_str:Union[str, List[str]], is_query:bool) -> torch.Tensor:
        if is_query:
            self.tokenizer.truncation_side = "left"
        else:
            self.tokenizer.truncation_side = "right"
        input_ids = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        embedding = self.model(**input_ids)[0][:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def retrieve(self, query_str:str, documents:List[CodeBlock], n:int=None, is_training:bool=True) \
        -> Union[torch.Tensor, List[CodeBlock]]:
        r'''
        Params:
            query_str:用于检索的字符串
            document_str:被检索的文档合集
            is_training:训练时返回各个document的分数；非训练时返回分数最高的n个document
        '''
        
        if not is_training:
            assert n is not None
            
            if len(documents) == 0:
                return []
            
            # 提供的文档数可能小于需要的文档数
            n = min(n, len(documents))
            
            self.eval()
            with torch.no_grad():
                query_embedding = self.embedding(query_str, True)
                document_embedding = self.embedding([doc.code_content for doc in documents], False)
                scores = torch.matmul(query_embedding, document_embedding.T).squeeze(dim=0)
            self.train()
            
            _, top_indices = torch.topk(scores, n)
            top_indices = top_indices.cpu().tolist()
            return [documents[i] for i in top_indices]
        
        else:
            assert len(documents) != 0
            
            query_embedding = self.embedding(query_str, True)
            document_embedding = self.embedding([doc.code_content for doc in documents], False)
            scores = torch.matmul(query_embedding, document_embedding.T).squeeze(dim=0)
        
            return scores
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
class UnixcoderForRetriever(Retriever):
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        
    def tokenize(self, input_str, is_query):
        tokens = self.tokenizer.tokenize(input_str)
        max_length = 512
        if is_query:
            tokens = tokens[-(max_length - 4):]
        else:
            tokens = tokens[:(max_length - 4)]
        tokens = [self.tokenizer.cls_token, "<encoder-only>", self.tokenizer.sep_token] \
            + tokens + [self.tokenizer.sep_token]
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_length - len(tokens_id)
        tokens_id += [self.tokenizer.pad_token_id] * padding_length
        return tokens_id
        
    def embedding(self, input_str, is_query):
        if not isinstance(input_str, str):
            source_ids = [self.tokenize(string, is_query) for string in input_str]
        else:
            source_ids = [self.tokenize(input_str, is_query)]
        
        source_ids = torch.tensor(source_ids, dtype=torch.long).to("cuda")
        mask = source_ids.ne(self.tokenizer.pad_token_id).to("cuda")
        token_embeddings = self.model(source_ids, attention_mask=mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings