from utils.util import (split_into_smaller_blocks, bm25_retrieve,
                        CodeBlock, Example)
from tqdm import tqdm
from multiprocessing import Pool
import random
import argparse
import pandas as pd
import numpy as np
import pickle
import os

# Global variables for multiprocessing
global_relevant_code_num = None
global_dataset_name = None
global_without_suffix_ratio = None

def init_pool(relevant_code_num, dataset_name, without_suffix_ratio):
    """Initialize global variables for each pool worker"""
    global global_relevant_code_num, global_dataset_name, global_without_suffix_ratio
    global_relevant_code_num = relevant_code_num
    global_dataset_name = dataset_name
    global_without_suffix_ratio = without_suffix_ratio

def process_item(item):
    """Process single item with global variables"""
    try:
        task_id, _, left_context, right_context, crossfile_context, groundtruth = item
        
        if global_without_suffix_ratio is not None and random.random() < global_without_suffix_ratio:
            right_context = ""
        
        # Process crossfile context
        cross_files = crossfile_context if len(crossfile_context) > 0 else [
            {'path': "", "text": "Don't need cross file context for completion"}]
        cross_files = [CodeBlock(x["path"], x["text"]) for x in cross_files]
        
        code_blocks = []
        for file in cross_files:
            code_blocks.extend(split_into_smaller_blocks(file, True))
        
        # Process context windows
        prefix_line = left_context.split("\n")
        prefix_part = prefix_line[-8:] if len(prefix_line) >= 8 else prefix_line
        remaining = 15 - len(prefix_part)
        suffix_line_clean = [line for line in right_context.split('\n') if line.strip()]
        suffix_part = suffix_line_clean[:remaining]
        query_str = "\n".join(prefix_part + suffix_part)
        
        # BM25 retrieval
        scores = bm25_retrieve(query_str, 
                             [cb.code_content for cb in code_blocks],
                             global_relevant_code_num)
        sorted_indices = np.argsort(scores)[::-1][:global_relevant_code_num]
        retrieved_codeblocks = [code_blocks[idx] for idx in sorted_indices]
        
        # Create Example object
        if global_dataset_name == "repoeval":
            return Example(task_id, left_context+"\n", right_context, groundtruth, retrieved_codeblocks)
        return Example(task_id, left_context, right_context, groundtruth, retrieved_codeblocks)
    except Exception as e:
        print(f"Error processing {task_id}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--without_suffix_ratio", type=float, default=None)
    args = parser.parse_args()

    # Load dataset
    dataset_paths = args.dataset_path.split(",") if "," in args.dataset_path else [args.dataset_path]
    datasets = []
    for path in dataset_paths:
        if path.endswith(".json"):
            try:
                datasets.append(pd.read_json(path))
            except ValueError as e:
                datasets.append(pd.read_json(path, lines=True))
        elif path.endswith(".parquet"):
            datasets.append(pd.read_parquet(path))
        else:
            raise RuntimeError("Unsupported file type")
    dataset = pd.concat(datasets)

    # Prepare items for multiprocessing
    items = dataset[["task_id", "path", "left_context", "right_context", 
                   "crossfile_context", "groundtruth"]].values.tolist()

    # Create process pool
    with Pool(processes=args.workers,
             initializer=init_pool,
             initargs=(args.relevant_code_num,
                      args.dataset_name,
                      args.without_suffix_ratio)) as pool:
        
        results = list(tqdm(pool.imap(process_item, items, chunksize=10),
                      total=len(items),
                      desc=f"Processing {args.dataset_name}"))
    
    # Filter failed items and save
    processed_dataset = [r for r in results if r is not None]
    
    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")
    
    output_path = os.path.join("preprocessed", 
                              f"{args.dataset_name}-{args.relevant_code_num}.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f)

if __name__ == "__main__":
    main()