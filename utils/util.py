import random
import os
import torch
import pickle
import numpy as np
import rank_bm25
from nltk import word_tokenize
from tqdm import tqdm
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
    
    def __eq__(self, value):
        assert type(value) == CodeBlock
        if self.file_path == value.file_path and self.code_content == value.code_content:
            return True
        else:
            return False
        
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 prefix_ids:List[int],
                 suffix_ids:List[int],
                 middle_ids:List[int] = None,
                 project_name_ids:List[int] = None,
                 query:str = None,
                 document:List[CodeBlock] = None,
    ):
        self.prefix_ids = prefix_ids
        self.middle_ids = middle_ids
        self.suffix_ids = suffix_ids
        self.project_name_ids = project_name_ids
        self.query = query
        self.document = document
        
class Benchmarks(dict):
    def __init__(self, test_datasets, tokenizer:AutoTokenizer, args):
        self.test_datasets = test_datasets
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
                    example.relevant_code, self.tokenizer, example.task_id))
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

def load_dataset(datasetname:str, k:int) -> List[Example]:
    """
    Loads a dataset.
    :param datasetname: The name of the dataset to load.
    :return: The loaded dataset.
    """
    file_name = f"{datasetname}-{k}.pkl"
    with open(f"preprocessed/{file_name}", 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset

def convert_example_to_feature(prefix:str, suffix:str, middle:str, related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, task_ids:str=None) -> InputFeatures:
    
    prefix_line = prefix.split("\n")
    # 处理前缀部分：最多取8行
    prefix_part = prefix_line[-8:] if len(prefix_line) >= 8 else prefix_line
    remaining = 15 - len(prefix_part)  # 计算需要从后缀补充的行数

    # 处理后缀部分：过滤空行并取剩余需要的行数
    suffix_line_clean = [line for line in suffix.split('\n') if line.strip()]
    suffix_part = suffix_line_clean[:remaining]

    # 合并结果
    query_str = "\n".join(prefix_part + suffix_part)
    
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    middle_tokenized_result = tokenizer(middle, add_special_tokens=False)
    
    if task_ids is not None:
        project_name = task_ids.split("/")[0]
        project_name_result = tokenizer(project_name, add_special_tokens=False)
    else:
        project_name_result = {
            "input_ids": None
        }

    feature = InputFeatures(
        prefix_tokenized_result["input_ids"],
        suffix_tokenized_result["input_ids"],
        middle_tokenized_result["input_ids"],
        project_name_result["input_ids"],
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
        for i in range(0, min(len(lines),5000), 7):
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

def split_sentence(code:str) -> List[str]:
    return word_tokenize(code)

def bm25_retrieve(query_str:str, candidate_str:List[str], k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [split_sentence(doc) for doc in candidate_str]
    bm25_model = rank_bm25.BM25Okapi(tokenized_corpus)
    query = split_sentence(query_str)
    doc_scores = bm25_model.get_scores(query)
    return doc_scores

def get_cross_file_context(related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int) -> Dict[str, List[int]]:
    filter_codeblocks = []
    
    assert len(related_codes) <= 1
    
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        # TODO: 真的需要把file path也附加上去吗
        if file_path != "" and file_path != "Unknown":
            filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    
    if len(filter_codeblocks) > 0:
        related_tokenized_result = tokenizer(filter_codeblocks, add_special_tokens=False)
    else:
        return repo_content
    
    repo_content["input_ids"] = related_tokenized_result["input_ids"][0][:cross_file_budget]
    repo_content["attention_mask"] = related_tokenized_result["attention_mask"][0][:cross_file_budget]
    
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