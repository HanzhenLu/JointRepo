import random
import pandas as pd
import tokenize
import io
import re
import os
from fastbm25 import fastbm25
from typing import Tuple, List, Dict
from torch import FloatTensor, LongTensor
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria

class Example:
    def __init__(self, task_id:str, prefix:str, suffix:str, middle:str, relevant_codes:List["CodeBlock"]) -> None:
        self.task_id = task_id
        self.prefix = prefix
        self.suffix = suffix
        self.middle = middle
        self.relevant_code = relevant_codes

def load_train_and_valid_dataset(dataset_path:str) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """
    Loads the dataset.
    :return: The training dataset and validation dataset.
    """
    training_datasets = []
    validation_datasets = []
    data_frame = pd.read_parquet(dataset_path)
    all_data = []
    temp_data = []
    for x in data_frame[["path", "content", "first"]].values:
        # get the value in "first" column and start of a new data if it is true
        if x[-1]:
            # end the last data
            if len(temp_data) > 1:
                all_data.append(temp_data)
            temp_data = []
        temp_data.append([x[0], x[1]])
    # append the last data
    if temp_data:
        all_data.append(temp_data)
    
    random.shuffle(all_data)
    training_datasets = all_data[:int(len(all_data) * 0.9)]
    validation_datasets = all_data[int(len(all_data) * 0.9):]

    return training_datasets, validation_datasets

def load_dataset(datasetname:str, tokenizer, args) -> List[Example]:
    """
    Loads a dataset.
    :param datasetname: The name of the dataset to load.
    :return: The loaded dataset.
    """
    if os.path.exists(datasetname):
        data_frame = pd.read_json(datasetname)
    elif datasetname == "cceval_python":
        data_frame = pd.read_parquet("data/cceval/python/test.parquet")
    elif datasetname == "repoeval_line":
        data_frame_parts = []
        data_frame_parts.append(pd.read_parquet("data/repoeval/line_level/test_0.parquet"))
        data_frame_parts.append(pd.read_parquet("data/repoeval/line_level/test_1.parquet"))
        data_frame = pd.concat(data_frame_parts)
    elif datasetname == "github_projects":
        data_frame = pd.read_json("data/github_projects/python/train.json")
    elif datasetname == "ours":
        data_frame = pd.read_parquet("data/ours/python/test.parquet")
    elif datasetname == "ours_suffix":
        data_frame = pd.read_parquet("data/ours/python/test_suffix.parquet")
    else:
        raise Exception("Unsupport dataset name")
    
    datasets = []
    for item in data_frame[["task_id", "path", "left_context", "right_context", "crossfile_context", "groundtruth"]].values:
        cross_files = item[4] if len(item[4]) > 0 else [{'path': "", "text": "Don't need cross file context for completion"}]
        cross_files = [CodeBlock(x["path"], x["text"]) for x in cross_files]
        code_blocks:List[CodeBlock] = []
        for file in cross_files:
            code_blocks.extend(split_into_smaller_blocks(file, args.enable_fixed_block))
        
        prefix_line = item[2].split("\n")
        # 处理前缀部分：最多取8行
        prefix_part = prefix_line[-8:] if len(prefix_line) >= 8 else prefix_line
        remaining = 15 - len(prefix_part)  # 计算需要从后缀补充的行数

        # 处理后缀部分：过滤空行并取剩余需要的行数
        suffix_line_clean = [line for line in item[3].split('\n') if line.strip()]
        suffix_part = suffix_line_clean[:remaining]

        # 合并结果
        query_str = "\n".join(prefix_part + suffix_part)
        result = bm25_retrieve(query_str, [code_block.code_content for code_block in code_blocks], tokenizer, args.relevant_code_num)
        retrieved_codeblocks = [code_blocks[x[1]] for x in result]
        
        if datasetname == "repoeval_line":
            datasets.append(Example(item[0], item[2]+"\n", item[3], item[5], retrieved_codeblocks))
        else:
            datasets.append(Example(item[0], item[2], item[3], item[5], retrieved_codeblocks))
    
    return datasets

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
        return self.code_content
    
def label_line(code:str) -> List[Tuple[List[int], bool]]:
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    
    for token_type, string, start, _, _ in tokens:
        # OP
        if token_type == 54:
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        # NL
        if token_type == 61 and len(stack) == 0:
            line_map.append(([start[0] - 1], False))
            line_count = start[0]
        
        # NEWLINE
        elif token_type == 4:
            line_map.append(([i - 1 for i in range(line_count+1, start[0]+1)], True))
            line_count = start[0]
    
    return line_map

def split_into_smaller_blocks(code_block:CodeBlock, enable_fixed_block:bool) -> List[CodeBlock]:
    """
    Split large blocks of code into smaller ones, each containing no more than 12 non-empty lines.
    """
    smaller_blocks = []

    # 每15行划分一个block
    if enable_fixed_block:
        lines = [line for line in code_block.code_content.split('\n') if line.strip() != '']
        for i in range(0, min(len(lines),5000), 8):
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

        # 超过12行的block划分成多个block
        max_len = 15
        temp_mini_blocks = []
        for mini_block in mini_blocks:
            if len(mini_block) > max_len:
                for idx in range(0, len(mini_block), max_len):
                    temp_mini_blocks.append(mini_block[idx: idx+max_len])
            else:
                temp_mini_blocks.append(mini_block)
        mini_blocks = temp_mini_blocks

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

# 模型生成的停止条件
class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int]) -> None:
        super().__init__()
        self.token_id_list = token_id_list
        
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs):
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

# 获得所有以\n结尾的token_id
def get_NL_list(tokenizer:AutoTokenizer) -> List[int]:
    NL_list = []
    for _, id in tokenizer.vocab.items():
        token = tokenizer.decode(id)
        if token.endswith("\n"):
            NL_list.append(id)
    return NL_list

def bm25_retrieve(query_str:str, candidate_str:List[str], tokenizer:PreTrainedTokenizer, k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [tokenizer.tokenize(doc)[:200] for doc in candidate_str]
    bm25_model = fastbm25(tokenized_corpus)
    query = tokenizer.tokenize(query_str)[:200]
    result = bm25_model.top_k_sentence(query, k=k)
    return result

def cross_file_contexts(related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int) -> Dict[str, List[int]]:
    filter_codeblocks = []
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        # TODO: 真的需要把file path也附加上去吗
        if file_path != "":
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