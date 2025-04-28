import os
import ast
import random
import pandas as pd
import io
import tokenize
import string
import argparse
import multiprocessing
import json
import rank_bm25
import numpy as np
import shutil
from functools import partial
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from functools import partial
from tqdm import tqdm
from utils.util import split_into_smaller_blocks, split_sentence, CodeBlock

# 定义允许的字符集合
ALLOWED = set(
    string.ascii_letters +    # 字母 a-zA-Z
    string.digits +           # 数字 0-9
    string.punctuation +      # 标点符号 !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    ' ' + '\n' + '\t'         # 空格、换行符、制表符
)

TEMP_DIR =  "/data/hanzhenlu/temp/intermediate/"

def label_line(code:str) -> List[Tuple[List[int], bool]]:
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    
    for token_type, string, start, _, _ in tokens:
        if token_type == tokenize.OP:
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        if token_type == tokenize.NL and len(stack) == 0:
            line_map.append(([start[0] - 1], False))
            line_count = start[0]
        
        elif token_type == tokenize.NEWLINE:
            line_map.append(([i - 1 for i in range(line_count+1, start[0]+1)], True))
            line_count = start[0]
    
    return line_map

def check_character_percentage(s: str) -> bool:
    """
    检查字符串中非允许字符的占比是否超过 10%
    
    Args:
        s (str): 需要检查的输入字符串
    
    Returns:
        bool: 非法字符占比 ≤10% 返回 True，否则返回 False
    """
    
    # 空字符串直接返回 True
    if not s:
        return True
    
    total_chars = len(s)
    
    # 统计非法字符数量
    invalid_count = sum(1 for char in s if char not in ALLOWED)
    
    # 计算非法字符占比（百分比）
    invalid_percent = (invalid_count / total_chars) * 100
    
    return invalid_percent <= 10

def check_character_exist(s:str) -> bool:
    for c in s:
        if c not in ALLOWED:
            return True
    return False

def read_dir(dir:str) -> Dict[str, str]:
    content_map = {}
    for dirpath, _, filenames in os.walk(dir, topdown=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                if code.strip() != "":
                    content_map[file_path] = code
            except:
                print(file_path)
            
    
    return content_map

def construct_data(code_content:str) -> Tuple[str, str, str]:
    if len(code_content.splitlines()) < 15 or not check_character_percentage(code_content):
        return None
    
    try:
        ast.parse(code_content)
    except Exception:
        return None
    line_index_labels = label_line(code_content)
    # if the selected file contains little content, we should drop it
    raw_lines = code_content.split('\n')
    raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
    valid_index = [r for r, v in line_index_labels if v]
    
    if len(valid_index) == 0:
        return None
    
    inner_try_count = 0
    
    selected_line = None
    while (selected_line is None or len(selected_line) > 200 or len(selected_line.strip()) < 10 or check_character_exist(selected_line)) and inner_try_count < 10:
        selected_index = random.choice(valid_index[1:]) if len(valid_index) > 1 else valid_index[0]
        # 清除行末的空格
        selected_line = "".join([raw_lines[i].rstrip() + "\n" for i in selected_index])
        inner_try_count += 1
    
    if inner_try_count == 10 or selected_line is None:
        return None
    
    prefix = "".join([raw_lines[i] for i in range(0, selected_index[0])]) if selected_index[0] > 0 else ""
    suffix = "".join([raw_lines[i] for i in range(selected_index[-1] + 1, len(raw_lines))]) if selected_index[-1] < len(raw_lines) else ""
    if random.random() > 0.8:
        start = 0
    else:
        start = random.randint(0, len(selected_line) - 10)
    if selected_line.count("\n") > 1 and random.random() > 0.5:
        newline_indices = [i for i, char in enumerate(selected_line) if char == "\n" and i > start]
        if len(newline_indices) > 0:
            end = random.choice(newline_indices) + 1
        else:
            end = len(selected_line)
    else:
        end = len(selected_line)
        
    prefix += selected_line[:start]
    middle = selected_line[start:end]
    suffix = selected_line[end:] + suffix
    
    return prefix, suffix, middle

def construct_data_repo(code_content:str) -> Tuple[str, str, str]:
    if len(code_content.splitlines()) < 15 or not check_character_percentage(code_content):
        return None
    
    try:
        ast.parse(code_content)
    except Exception:
        return None
    
    raw_lines = code_content.splitlines(keepends=True)
    valid_line_index = [i for i, line in enumerate(raw_lines) if len(line.strip()) > 15]
    if len(valid_line_index) == 0:
        return None
    selected_index = random.choice(valid_line_index)
    
    prefix = "".join(raw_lines[:selected_index])
    middle = raw_lines[selected_index].rstrip() + "\n"
    suffix = "".join(raw_lines[selected_index+1:])
    
    return prefix, suffix, middle
    

def process_repository(repo_name, construct_data_fn):
    """处理单个仓库的独立函数"""
    with open(os.path.join(TEMP_DIR, repo_name), 'r') as f:
        file_content_map = json.loads(f.read())
    
    file_names = list(file_content_map.keys())
    
    if not file_names:
        return []
    
    # 代码块分割逻辑
    code_blocks = []
    for path, text in file_content_map.items():
        file = CodeBlock(path, text)
        code_blocks.extend(split_into_smaller_blocks(file, False))
    
    # BM25模型构建
    corpus = [x.code_content for x in code_blocks]
    tokenized_corpus = [split_sentence(doc) for doc in corpus]
    bm25_model = rank_bm25.BM25Okapi(tokenized_corpus)
    
    # 样本生成
    samples = []
    # 打乱顺序，确保生成的数据多样性
    random.shuffle(file_names)
    for file_name in file_names:
        result = construct_data_fn(file_content_map[file_name])
        if not result:
            continue
            
        prefix, suffix, middle = result
        sample = {
            "path": file_name,
            "left_context": prefix,
            "right_context": suffix,
            "groundtruth": middle,
        }
        
        # 查询构建逻辑
        prefix_part = prefix.split("\n")[-8:]
        suffix_clean = [line for line in suffix.split('\n') if line.strip()]
        remaining = 15 - len(prefix_part)
        suffix_part = suffix_clean[:remaining]
        
        # 增加检索到有用信息的概率
        if random.random() > 0.5:
            query_str = "\n".join(prefix_part + suffix_part)
        else:
            query_str = "\n".join(prefix_part) + middle + "\n".join(suffix_part)
        
        # BM25检索
        query = split_sentence(query_str)
        doc_scores = bm25_model.get_scores(query)
        sorted_indices = np.argsort(doc_scores)[::-1][:50]
        
        related_codes = [
            code_blocks[idx] for idx in sorted_indices
            if code_blocks[idx].file_path != file_name
        ]
        
        sample["crossfile_context"] = [
            {"path": x.file_path, "text": x.code_content} 
            for x in related_codes
        ]
        
        samples.append(sample)
        if len(samples) >= 10:
            break
    
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", default="self-collected", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--repocoder", action="store_true")
    
    args = parser.parse_args()
    
    if args.dataset_name == "self-collected":
        root = "github_projects"
        repo_names = os.listdir(root)
        repo_dict:Dict[str, Dict[str, str]] = {}
        for repo_name in repo_names:
            repo_dict[repo_name] = read_dir(os.path.join(root, repo_name))
    elif args.dataset_name == "huawei":
        raw_data = pd.read_csv("/nasdata/datasets/huawei_datasets/python_dataset.csv")
        repo_dict = defaultdict(dict)
        base_path = "/data/jiaweilu/repos/"
        for full_path, code_content in zip(raw_data["Path"], raw_data["Code"]):
            # 校验路径格式
            if not full_path.startswith(base_path):
                print(f"警告: 路径 {full_path} 不符合基准路径格式，已跳过")
                continue

            # 提取项目名和相对路径
            relative_path = full_path[len(base_path):]
            path_obj = Path(relative_path)
            
            # 确保路径包含至少两部分（项目名 + 文件路径）
            if len(path_obj.parts) < 2:
                print(f"警告: 路径 {full_path} 结构不完整，已跳过")
                continue

            # 分割项目名和文件路径
            repo_name = path_obj.parts[0]
            file_path = str(Path(*path_obj.parts[0:]))

            # 存储到字典
            if file_path in repo_dict[repo_name]:
                print(f"警告: {repo_name} 中存在重复文件路径 {file_path}")
            repo_dict[repo_name][file_path] = code_content
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    for key, value in repo_dict.items():
        with open(os.path.join(TEMP_DIR, key), 'w') as f:
            value_str = json.dumps(value)
            f.write(value_str)
    
    keys = list(repo_dict.keys())
    del(repo_dict)
    del(raw_data)
    if args.debug:
        random.shuffle(keys)
        keys = keys[:len(keys) // 10]
    dataframe = {"task_id":[], "path":[], "left_context":[], "right_context":[], "crossfile_context":[], "groundtruth":[]}
    
    if args.repocoder:
        process_fn = partial(process_repository, construct_data_fn=construct_data_repo)
    else:
        process_fn = partial(process_repository, construct_data_fn=construct_data)
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_fn, keys),
            total=len(keys),
            desc="Processing repositories"
        ))
    
    task_counter = defaultdict(int)
    for repo_samples in results:
        if not repo_samples:
            continue
            
        repo_name = repo_samples[0]["path"].split('/')[0]  # 假设路径包含仓库名
        for sample in repo_samples:
            task_counter[repo_name] += 1
            dataframe["task_id"].append(f"{repo_name}/{task_counter[repo_name]}")
            for key in ["path", "left_context", "right_context", 
                       "crossfile_context", "groundtruth"]:
                dataframe[key].append(sample[key])
    
    dataframe = pd.DataFrame.from_dict(dataframe)
    if not os.path.exists("data/github_projects/python"):
        os.makedirs("data/github_projects/python")
    if args.repocoder:
        dataframe.to_json(f"data/github_projects/python/{args.dataset_name}-repocoder_4_15.json")
    else:
        dataframe.to_json(f"data/github_projects/python/{args.dataset_name}-train_4_18.json")
    
    shutil.rmtree(TEMP_DIR)