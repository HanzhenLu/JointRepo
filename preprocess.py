from transformers import AutoTokenizer
from utils.util import (split_into_smaller_blocks, bm25_retrieve,
                        CodeBlock, List, Example)
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import pickle
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    
    args = parser.parse_args()
    dataset_path:str = args.dataset_path
    tokenizer_name = Path(args.tokenizer_path).parts[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    if "," in dataset_path:
        dataset_paths = dataset_path.split(",")
    else:
        dataset_paths = [dataset_path]
        
    datasets = []
    for path in dataset_paths:
        if dataset_path.endswith(".json"):
            dataset = pd.read_json(path)
        elif dataset_path.endswith(".parquet"):
            dataset = pd.read_parquet(path)
        else:
            raise RuntimeError("unsupported file type")
        datasets.append(dataset)
    dataset = pd.concat(datasets)
    
    
    processed_dataset = []
    for item in tqdm(dataset[["task_id", "path", "left_context", "right_context", "crossfile_context", "groundtruth"]].values, desc=f"processing {args.dataset_name}"):
        cross_files = item[4] if len(item[4]) > 0 else [{'path': "", "text": "Don't need cross file context for completion"}]
        cross_files = [CodeBlock(x["path"], x["text"]) for x in cross_files]
        code_blocks:List[CodeBlock] = []
        for file in cross_files:
            code_blocks.extend(split_into_smaller_blocks(file, True))
        
        prefix_line = item[2].split("\n")
        # 处理前缀部分：最多取8行
        prefix_part = prefix_line[-8:] if len(prefix_line) >= 8 else prefix_line
        remaining = 15 - len(prefix_part)  # 计算需要从后缀补充的行数

        # 处理后缀部分：过滤空行并取剩余需要的行数
        suffix_line_clean = [line for line in item[3].split('\n') if line.strip()]
        suffix_part = suffix_line_clean[:remaining]

        # 合并结果
        query_str = "\n".join(prefix_part + suffix_part)
        scores = bm25_retrieve(query_str, [code_block.code_content for code_block in code_blocks], tokenizer, args.relevant_code_num)
        sorted_indices = np.argsort(scores)[::-1][:args.relevant_code_num]
        retrieved_codeblocks = [code_blocks[idx] for idx in sorted_indices]
        
        if "repoeval" == args.dataset_name:
            processed_dataset.append(Example(item[0], item[2]+"\n", item[3], item[5], retrieved_codeblocks))
        else:
            processed_dataset.append(Example(item[0], item[2], item[3], item[5], retrieved_codeblocks))
    
    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")
    with open(os.path.join("preprocessed", f"{args.dataset_name}-{tokenizer_name}-{args.relevant_code_num}.pkl"), 'wb') as f:
        pickle.dump(processed_dataset, f)

if __name__ == "__main__":
    main()