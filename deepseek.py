from __future__ import absolute_import
import os
import torch
import random
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
from fastbm25 import fastbm25
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (PreTrainedTokenizer, PreTrainedModel)

from utils.util import (load_dataset, CodeBlock, split_into_smaller_blocks, Example)
from model import build_model
from eval import compute_metric_stmt

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids:List[int],
                 attention_mask:List[int],
                 target_ids:List[int]
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target_ids = target_ids
  
class CustomDataset(Dataset):
    """A dataset class for code generation

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_codeblocks: Retrieved code blocks.
    """
    
    def __init__(self, args, tokenizer, examples) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.special_tokens = {
            "prefix_id": tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>"),
            "suffix_id": tokenizer.convert_tokens_to_ids("<｜fim▁end｜>"),
            "middle_id": tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>")
        }
        
    def __len__(self):
        return len(self.examples)
    
    def construct_prompt(self, example:Example) -> Dict[str, List[int]]:
        '''
        concatenate the context from other files with the context within the current file
        '''
        prefix = example.prefix
        suffix = example.suffix
        
        code_blocks:List[CodeBlock] = []
        for file in example.relevant_code:
            code_blocks.extend(split_into_smaller_blocks(file, self.args.enable_fixed_block))
            
        tokenized_corpus = [self.tokenizer.tokenize(doc.code_content) for doc in code_blocks]
        bm25_model = fastbm25(tokenized_corpus)
        prefix_line = prefix.split("\n")
        # TODO: set the query_line as a hyperparameter
        if len(prefix_line) >= 15:
            query = "\n".join(prefix_line[-15:])
        else:
            query = "\n".join(prefix_line)
            suffix_line = suffix.split("\n")
            query += "\n" + "\n".join(suffix_line[:15-len(prefix_line)])
        query = self.tokenizer.tokenize(query)
        result = bm25_model.top_k_sentence(query, k=self.args.relevant_code_num)
        retrieved_codeblocks = [code_blocks[x[1]] for x in result]
        
        filter_codeblocks = []
        for x in retrieved_codeblocks:
            file_path = x.file_path
            code_content = x.code_content
            if file_path != "":
                filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
            else:
                break
        
        prefix_tokenized_result = self.tokenizer(prefix, add_special_tokens=False)
        suffix_tokenized_result = self.tokenizer(suffix, add_special_tokens=False)
        if len(filter_codeblocks) > 0:
            related_tokenized_result = self.tokenizer(filter_codeblocks, add_special_tokens=False)
        
        # TODO: set the cross_file_budget as a hyperparameter
        cross_file_budget = int(0.25 * self.args.max_input_length)
        
        repo_content = {
            "input_ids": [],
            "attention_mask": []
        }
        related_idx = 0
        while related_idx < len(filter_codeblocks) and len(repo_content["attention_mask"]) + len(related_tokenized_result["input_ids"][related_idx]) < cross_file_budget:
            repo_content["input_ids"].extend(related_tokenized_result["input_ids"][related_idx])
            repo_content["attention_mask"].extend(related_tokenized_result["attention_mask"][related_idx])
            related_idx += 1
        
        left_budget = self.args.max_input_length - len(repo_content["input_ids"]) - 4
        prefix_length = int(left_budget / 2)
        suffix_length = int(left_budget - prefix_length)
        prefix_ids = prefix_tokenized_result["input_ids"] if len(prefix_tokenized_result["input_ids"]) < prefix_length else prefix_tokenized_result["input_ids"][-prefix_length:]
        suffix_ids = suffix_tokenized_result["input_ids"] if len(suffix_tokenized_result["input_ids"]) < suffix_length else suffix_tokenized_result["input_ids"][:suffix_length]
        
        direct_content = {
            # TODO: prefix id 放在哪需要进一步测试
            "input_ids": [self.special_tokens["prefix_id"]] + prefix_ids + [self.special_tokens["middle_id"]] + suffix_ids + [self.special_tokens["suffix_id"]],
            "attention_mask": [1] * (len(prefix_ids) + len(suffix_ids) + 3)
        } 
        
        input_ids = [32013] + repo_content["input_ids"] + direct_content["input_ids"]
        attention_mask = [1] + repo_content["attention_mask"] + direct_content["attention_mask"]
        padding_length = self.args.max_input_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        attention_mask = [0] * padding_length + attention_mask
        
        assert len(input_ids) == len(attention_mask) == self.args.max_input_length
        
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask
        }
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        input_ids, attention_mask = self.construct_prompt(example).values()
        return torch.tensor(input_ids), torch.tensor(attention_mask)

def generate(args, generator:PreTrainedModel, tokenizer:PreTrainedTokenizer, examples:List[Example]) -> List[str]:
    generated_codes = []
    dataset = CustomDataset(args, tokenizer, examples)
    sampler = SequentialSampler(dataset)
    # TODO: set the num_workers as a hyperparameter
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=20)
    pbar = tqdm(dataloader, desc="Generating")
    with torch.no_grad():
        for batch in pbar:
            input_ids, attention_mask = [x.to(generator.device) for x in batch]
            generated_ids = generator.generate(input_ids, attention_mask=attention_mask, 
                max_length=input_ids.size(1)+args.max_generation_length, pad_token_id=tokenizer.pad_token_id)
            generated_codes.append(generated_ids[:, input_ids.size(1):])
    generated_codes = torch.cat(generated_codes, 0)
    return [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]

def test(all_eval_examples:Dict[str, List[Example]], generator:PreTrainedModel, tokenizer:PreTrainedTokenizer, args, epoch:int):
    for name, examples in all_eval_examples.items():
        print("Evaluating on {} dataset".format(name))
        generations = generate(args, generator, tokenizer, examples)
        if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
            os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
        with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
            for example, generation in zip(examples, generations):
                f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")
            if name == "cceval_python":
                results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl")
            elif name == "repoeval_line":
                results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl")
        
        print(f"{name} epoch {epoch}: {str(results)}")
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
    
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--GPU_ids', type=str, default='0',
                        help="The ids of GPUs will be used")
    
    parser.add_argument('--enable_fixed_block', action='store_true',
                        help="Split code into blocks by fixed line")
    parser.add_argument('--relevant_code_num', default=3, type=int,
                        help="Total number of relevant code blocks to use")
    parser.add_argument('--max_input_length', default=1024, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_generation_length', default=30, type=int,
                        help="Max token num for generating when evaluate")
    
    # print arguments
    args = parser.parse_args()
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    logger_path = os.path.join(args.output_dir, 'test')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    # build model
    generator, tokenizer = build_model(args)
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)  
    
    all_eval_examples = {
        "cceval_python": load_dataset("cceval_python"),
        "repoeval_line": load_dataset("repoeval_line")
    }
    
    test(all_eval_examples, generator, tokenizer, args, epoch=0)
                
if __name__ == '__main__':
    main()  
