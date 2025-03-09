import random
import pandas as pd
import re
import os
import torch
import numpy as np
from tqdm import tqdm
from fastbm25 import fastbm25
from typing import List, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer

class Example:
    def __init__(self, task_id:str, prefix:str, suffix:str, middle:str, relevant_codes:List["CodeBlock"]) -> None:
        self.task_id = task_id
        self.prefix = prefix
        self.suffix = suffix
        self.middle = middle
        self.relevant_code = relevant_codes

class CodeBlock(object):
    def __init__(self, file_path:str, code_content:str):
        """
        Represents a block of code.
        :param file_path: The path to the code file.
        :param code_content: The content of the code block.
        """
        self.file_path:str = file_path
        self.code_content:str = code_content

    def __str__(self):
        return f"#{self.file_path}\n{self.code_content}"
        
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 prefix_ids:List[int],
                 suffix_ids:List[int],
                 middle_ids:List[int] = None,
                 query:str = None,
                 document:List[CodeBlock] = None,
    ):
        self.prefix_ids = prefix_ids
        self.middle_ids = middle_ids
        self.suffix_ids = suffix_ids
        self.query = query
        self.document = document
        
class Benchmarks(dict):
    def __init__(self, valid_dataset:List[Example], tokenizer:AutoTokenizer, args):
        self.test_datasets = {
            "github_projects": valid_dataset,
            "ours_suffix": load_dataset("ours_suffix"),
            "ours": load_dataset("ours"),
            "cceval_python": load_dataset("cceval_python"),
            "repoeval_line": load_dataset("repoeval_line")
        }
        self.test_features = {}
        self.tokenizer = tokenizer
        self.args = args
    
    def __getitem__(self, key):
        if key not in self.test_datasets.keys():
            return None
        if key not in self.test_features:
            features = []
            for example in tqdm(self.test_datasets[key], desc=f"convert {key} into features"):
                example:Example
                features.append(convert_example_to_feature(example.prefix, example.suffix, example.middle, \
                    example.relevant_code, self.tokenizer, self.args))
            self.test_features[key] = features
        else:
            features = self.test_features[key]
        return features
        
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_dataset(datasetname:str) -> List[Example]:
    """
    Loads a dataset.
    :param datasetname: The name of the dataset to load.
    :return: The loaded dataset.
    """
    if datasetname == "cceval_python":
        data_frame = pd.read_parquet("data/cceval/python/test.parquet")
    elif datasetname == "repoeval_line":
        data_frame_parts = []
        data_frame_parts.append(pd.read_parquet("data/repoeval/line_level/test_0.parquet"))
        data_frame_parts.append(pd.read_parquet("data/repoeval/line_level/test_1.parquet"))
        data_frame = pd.concat(data_frame_parts)
    elif datasetname == "github_projects":
        data_frame = pd.read_json("data/github_projects/python/train.json")
    elif datasetname == "github_repos":
        data_frame = pd.read_parquet("data/github_repos/python/train.parquet")
    elif datasetname == "ours":
        data_frame = pd.read_parquet("data/ours/python/test.parquet")
    elif datasetname == "ours_suffix":
        data_frame = pd.read_parquet("data/ours/python/test_suffix.parquet")
    else:
        raise Exception("Unsupport dataset name")

    datasets = []
    for item in data_frame[["task_id", "path", "left_context", "right_context", "crossfile_context", "groundtruth"]].values:
        cross_files = item[4]
        cross_files = [CodeBlock(x["path"], x["text"]) for x in cross_files]
        if datasetname == "repoeval_line":
            datasets.append(Example(item[0], item[2]+"\n", item[3], item[5], cross_files))
        else:
            datasets.append(Example(item[0], item[2], item[3], item[5], cross_files))
    
    return datasets

def convert_example_to_feature(prefix:str, suffix:str, middle:str, related_files:List[CodeBlock], tokenizer:PreTrainedTokenizer, args) -> InputFeatures:
    
    code_blocks:List[CodeBlock] = []
    for file in related_files:
        code_blocks.extend(split_into_smaller_blocks(file, args.enable_fixed_block))
    
    candidate_str = [x.code_content for x in code_blocks]
    prefix_line = prefix.split("\n")
    # TODO: set the query_line as a hyperparameter
    if len(prefix_line) >= 15:
        query_str = "\n".join(prefix_line[-15:])
    else:
        query_str = "\n".join(prefix_line)
        suffix_line = suffix.split("\n")
        query_str += "\n" + "\n".join(suffix_line[:15-len(prefix_line)])
    
    candidate_num = args.relevant_code_num * 5
    result = bm25_retrieve(query_str, candidate_str, tokenizer, k=candidate_num)
    
    related_codes = [code_blocks[x[1]] for x in result]
    for i in range(args.relevant_code_num):
        related_codes.append(CodeBlock("Unknown", f"# Don't need crossfile {i}"))
    
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    middle_tokenized_result = tokenizer(middle, add_special_tokens=False)

    feature = InputFeatures(
        prefix_tokenized_result["input_ids"],
        suffix_tokenized_result["input_ids"],
        middle_tokenized_result["input_ids"],
        query_str,
        related_codes
    )
    return feature

def split_into_smaller_blocks(code_block:CodeBlock, enable_fixed_block:bool) -> List[CodeBlock]:
    """
    Split large blocks of code into smaller ones, each containing no more than 12 non-empty lines.
    """
    smaller_blocks = []

    # 每15行划分一个block
    if enable_fixed_block:
        lines = [line for line in code_block.code_content.split('\n') if line.strip() != '']
        for i in range(0, min(len(lines),5000), 15):
            start_line_offset = i
            end_line_offset = min(i + 15, len(lines))
            block_content = '\n'.join(lines[start_line_offset:end_line_offset])
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            block_content))

    else:
        # Split the code by spaces, then reassemble it into blocks.
        mini_blocks = []
        current_block = [] 
        for line in code_block.code_content.splitlines(): 
            if line.strip() == '':  
                if current_block: 
                    mini_blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)
        if current_block: 
            mini_blocks.append(current_block)

        # 超过15行的block划分成多个block
        max_len = 15
        temp_mini_blocks = []
        for mini_block in mini_blocks:
            if len(mini_block) > max_len:
                for idx in range(0, len(mini_block), max_len):
                    temp_mini_blocks.append(mini_block[idx: idx+max_len])
            else:
                temp_mini_blocks.append(mini_block)
        mini_blocks = temp_mini_blocks

        # combine two code block into a larger code block
        current_content = []
        total_lines = 0  
        for block in mini_blocks:
            if total_lines >= 5000:  
                break  
            if len(current_content) + len(block) <= max_len:  
                current_content.extend(block)
                total_lines += len(block)  
            else:  
                if current_content:  
                    smaller_blocks.append(CodeBlock(code_block.file_path, 
                                                    '\n'.join(current_content)))
                current_content = block  
                total_lines += len(block)  
        if current_content:  
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            '\n'.join(current_content)))
        
    return smaller_blocks

def split_word(word:str) -> List[str]:
    words = []
    
    if len(word) <= 1:
        return word

    word_parts = re.split('[^0-9a-zA-Z]', word)
    for part in word_parts:
        part_len = len(part)
        if part_len == 1:
            words.append(part)
            continue
        word = ''
        for index, char in enumerate(part):
            # condition : level|A
            if index == part_len - 1 and char.isupper() and part[index-1].islower():
                if word != '':
                    words.append(word)
                words.append(char)
                word = ''
                
            elif(index != 0 and index != part_len - 1 and char.isupper()):
                # condition 1 : FIRST|Name
                # condition 2 : first|Name
                condition1 = part[index-1].isalpha() and part[index+1].islower()
                condition2 = part[index-1].islower() and part[index+1].isalpha()
                if condition1 or condition2:
                    if word != '':
                        words.append(word)
                    word = char
                else:
                    word += char
            
            else:
                word += char
        
        if word != '':
            words.append(word)
            
    return [word.lower() for word in words]

def bm25_retrieve(query_str:str, candidate_str:List[str], tokenizer:PreTrainedTokenizer, k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [tokenizer.tokenize(doc)[:200] for doc in candidate_str]
    bm25_model = fastbm25(tokenized_corpus)
    query = tokenizer.tokenize(query_str)[:200]
    result = bm25_model.top_k_sentence(query, k=k)
    return result

def get_cross_file_context(related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int) -> Dict[str, List[int]]:
    filter_codeblocks = []
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        # TODO: 真的需要把file path也附加上去吗
        if file_path != "" and file_path != "Unknown":
            filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    if len(filter_codeblocks) > 0:
        related_tokenized_result = tokenizer(filter_codeblocks, add_special_tokens=False)
    
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    related_idx = 0
    while related_idx < len(filter_codeblocks) and len(repo_content["input_ids"]) + len(related_tokenized_result["input_ids"][related_idx]) < cross_file_budget:
        repo_content["input_ids"].extend(related_tokenized_result["input_ids"][related_idx])
        repo_content["attention_mask"].extend(related_tokenized_result["attention_mask"][related_idx])
        related_idx += 1
    
    return repo_content

def check_memory():
    # 获取当前 GPU 的显存占用（单位：字节）
    allocated = torch.cuda.memory_allocated()  # 当前分配的显存
    reserved = torch.cuda.memory_reserved()    # 缓存分配的显存（PyTorch 内部管理）

    # 转换为更友好的单位（如 MB）
    allocated_mb = allocated / 1024**3
    reserved_mb = reserved / 1024**3

    print(f"当前显存占用: {allocated_mb:.2f} GB")
    print(f"PyTorch 保留的显存: {reserved_mb:.2f} GB")