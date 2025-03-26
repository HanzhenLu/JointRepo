from __future__ import absolute_import
import os
import torch
import random
import logging
import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import (PreTrainedTokenizer, PreTrainedModel)

from utils.util import (load_dataset,
                        CodeBlock, Example)
from model import (build_model, generate)
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

class MyDataset(Dataset):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]
    
def DoNothingCollator(batch):
    return batch

def test(all_eval_examples:Dict[str, List[Example]], generator:PreTrainedModel, tokenizer:PreTrainedTokenizer, args, epoch:int):
    generator.eval()
    for name, examples in all_eval_examples.items():
        logger.info("Evaluating on {} dataset".format(name))
        model_to_generate = generator.module if hasattr(generator, "module") else generator
        generations = generate(args, model_to_generate, tokenizer, examples)
        if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
            os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
        with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
            for example, generation in zip(examples, generations):
                f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")
        if name == "cceval_python":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl")
        elif name == "repoeval_line":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl")
        elif name == "ours":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/ours/python/test.jsonl")
        elif name == "ours_suffix":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/ours/python/test_suffix.jsonl")
        elif name == "github_projects":
            targets, generations = ["".join(x.middle.split()) for x in examples], ["".join(x.split()) for x in generations]
            results = {}
            results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
        else:
            raise Exception("unsupport test set")
        logger.info(f"{name} epoch {epoch}: {str(results)}")
    if hasattr(generator, "module"):
        model_to_save = generator.module
    else:
        model_to_save = generator
    torch.save(model_to_save.state_dict(), f"{args.output_dir}/result_{epoch}/model.pth")
    generator.train()

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
    
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--GPU_ids', type=str, default='0',
                        help="The ids of GPUs will be used")
    
    parser.add_argument('--enable_fixed_block', action='store_true',
                        help="Split code into blocks by fixed line")
    parser.add_argument('--relevant_code_num', default=5, type=int,
                        help="Total number of relevant code blocks to use")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_generation_length', default=50, type=int,
                        help="Max token num for generating when evaluate")
    
    # print arguments
    args = parser.parse_args()
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    logger_path = os.path.join(args.output_dir, 'test.log')
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
    set_seed()

    # build model
    generator, tokenizer = build_model(args)
    tokenizer_name = Path(args.model_name_or_path).parts[-1]
    
    all_eval_examples = {}
    with open("preprocessed/cceval-opc-sft-v1-1.pkl", 'rb') as f:
        all_eval_examples["cceval_python"] = pickle.load(f)
    with open("preprocessed/ours-opc-sft-v1-1.pkl", 'rb') as f:
        all_eval_examples["ours"] = pickle.load(f)
    with open("preprocessed/ours-suffix-opc-sft-v1-1.pkl", 'rb') as f:
        all_eval_examples["ours_suffix"] = pickle.load(f)
    with open("preprocessed/repoeval-opc-sft-v1-1.pkl", 'rb') as f:
        all_eval_examples["repoeval_line"] = pickle.load(f)
    
    with open(f"{args.output_dir}/retrieval.pkl", 'wb') as f:
        pickle.dump(all_eval_examples, f)
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)
    
    eval_examples_sorted = []
    for i in range(args.relevant_code_num + 1):
        current_eval_examples:Dict[str, List[Example]] = {}
        for key, value in all_eval_examples.items():
            current_list:List[Example] = []
            for example in value:
                current_list.append(Example(example.task_id, example.prefix, 
                                                          example.suffix, example.middle, 
                                                          [example.relevant_code[i]] if len(example.relevant_code) > i 
                                                          else [CodeBlock("", "")]))
            current_eval_examples[key] = current_list
        eval_examples_sorted.append(current_eval_examples)
    
    for i, eval_examples in enumerate(eval_examples_sorted):
        test(eval_examples, generator, tokenizer, args, i)
                
if __name__ == '__main__':
    main()
