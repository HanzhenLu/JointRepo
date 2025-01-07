import os
import ast
import random
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from fastbm25 import fastbm25

from utils.util import label_line, split_into_smaller_blocks, CodeBlock

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
    while (selected_line is None or len(selected_line) > 200) and inner_try_count < 10:
        selected_index = random.choice(valid_index[1:]) if len(valid_index) > 1 else valid_index[0]
        selected_line = "".join([raw_lines[i] for i in selected_index])
        inner_try_count += 1
    
    if inner_try_count == 10 or selected_line is None:
        return None
    
    prefix = "".join([raw_lines[i] for i in range(0, selected_index[0])]) if selected_index[0] > 0 else ""
    suffix = "".join([raw_lines[i] for i in range(selected_index[-1] + 1, len(raw_lines))]) if selected_index[-1] < len(raw_lines) else ""
    if random.random() > 0.9:
        middle = selected_line
    else:
        split_point = random.randint(0, len(selected_line) - 2)
        prefix += selected_line[:split_point]
        middle = selected_line[split_point:]
    
    
    return prefix, suffix, middle

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("/data/hanzhenlu/LLaMA-Factory/saves/llama_100m_SPM_pretrained_sft_v4")
    
    root = "github_projects"
    repo_names = os.listdir(root)
    repo_dict:Dict[str, Dict[str, str]] = {}
    for repo_name in repo_names:
        repo_dict[repo_name] = read_dir(os.path.join(root, repo_name))

    dataframe = {"task_id":[], "path":[], "left_context":[], "right_context":[], "crossfile_context":[], "groundtruth":[]}
    for repo_name, file_content_map in tqdm(repo_dict.items()):
        file_names = file_content_map.keys()
        if len(file_names) == 0:
            continue
        count = 0
        samples = []
        
        code_blocks:List[CodeBlock] = []
        for path, text in file_content_map.items():
            file = CodeBlock(path, text)
            code_blocks.extend(split_into_smaller_blocks(file, False))
            
        corpus = [x.code_content for x in code_blocks]
        tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
        bm25_model = fastbm25(tokenized_corpus)
        
        for file_name in file_names:
            result = construct_data(file_content_map[file_name])
            if result is not None:
                prefix, suffix, middle = result
                sample = {}
                sample["path"] = file_name
                sample["left_context"] = prefix
                sample["right_context"] = suffix
                sample["groundtruth"] = middle
                
                prefix_line = prefix.split("\n")
                if len(prefix_line) >= 15:
                    query = "\n".join(prefix_line[-15:])
                else:
                    query = "\n".join(prefix_line)
                    suffix_line = suffix.split("\n")
                    query += "\n" + "\n".join(suffix_line[:15-len(prefix_line)])
                query = tokenizer.tokenize(query)
                result = bm25_model.top_k_sentence(query, k=50)
                related_codes = [code_blocks[x[1]] for x in result if code_blocks[x[1]].file_path != file_name]
                
                sample["crossfile_context"] = [
                    {
                        "path": x.file_path,
                        "text": x.code_content
                    } for x in related_codes
                ]
                
                samples.append(sample)
        
        if len(samples) > 200:
            selected_samples = random.choices(samples, k=200)
        else:
            selected_samples = samples
        
        for sample in selected_samples:
            count += 1
            dataframe["crossfile_context"].append(sample["crossfile_context"])
            dataframe["groundtruth"].append(sample["groundtruth"])
            dataframe["left_context"].append(sample["left_context"])
            dataframe["path"].append(sample["path"])
            dataframe["right_context"].append(sample["right_context"])
            dataframe["task_id"].append(f"{repo_name}/{count}")
    
    dataframe = pd.DataFrame.from_dict(dataframe)
    if not os.path.exists("data/github_projects/python"):
        os.makedirs("data/github_projects/python")
    dataframe.to_json("data/github_projects/python/train.json")