from __future__ import absolute_import
import os
import torch
import random
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, PreTrainedModel)

from utils.util import (load_dataset, CodeBlock, split_into_smaller_blocks, Example, bm25_retrieve)
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
        
def convert_example_to_feature(prefix:str, suffix:str, middle:str, related_files:List[CodeBlock], tokenizer:PreTrainedTokenizer, args) -> InputFeatures:
    prefix_id = tokenizer.convert_tokens_to_ids("<PREFIX>")
    suffix_id = tokenizer.convert_tokens_to_ids("<SUFFIX>")
    middle_id = tokenizer.convert_tokens_to_ids("<MIDDLE>")
    
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
    result = bm25_retrieve(query_str, candidate_str, k=args.relevant_code_num)
    
    related_codes = [code_blocks[x[1]] for x in result]
    filter_codeblocks = []
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        if file_path != "":
            filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    if len(related_codes) > 0:
        related_tokenized_result = tokenizer(filter_codeblocks, add_special_tokens=False)
        
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    middle_tokenized_result = tokenizer(middle, add_special_tokens=False)
    
    # TODO: set the cross_file_budget as a hyperparameter
    cross_file_budget = int(0.25 * args.max_input_length)
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    related_idx = 0
    while related_idx < len(related_codes) and len(repo_content["input_ids"]) + len(related_tokenized_result["input_ids"][related_idx]) < cross_file_budget:
        repo_content["input_ids"].extend(related_tokenized_result["input_ids"][related_idx])
        repo_content["attention_mask"].extend(related_tokenized_result["attention_mask"][related_idx])
        related_idx += 1
    
    left_budget = args.max_input_length - len(repo_content["input_ids"]) - len(middle_tokenized_result["input_ids"]) - 3
    prefix_length = int(left_budget / 2)
    suffix_length = int(left_budget - prefix_length)
    prefix_ids = prefix_tokenized_result["input_ids"] if len(prefix_tokenized_result["input_ids"]) < prefix_length else prefix_tokenized_result["input_ids"][-prefix_length:]
    suffix_ids = suffix_tokenized_result["input_ids"] if len(suffix_tokenized_result["input_ids"]) < suffix_length else suffix_tokenized_result["input_ids"][:suffix_length]
    middle_ids = middle_tokenized_result["input_ids"]
    direct_content = {
        "input_ids": [suffix_id] + suffix_ids + [prefix_id] + prefix_ids + [middle_id] + middle_ids,
        "attention_mask": [1] * (len(prefix_ids) + len(suffix_ids) + len(middle_ids) + 3)
    }
    
    
    input_ids = repo_content["input_ids"] + direct_content["input_ids"]
    attention_mask = repo_content["attention_mask"] + direct_content["attention_mask"]
    feature = InputFeatures(input_ids + [tokenizer.pad_token_id] * (args.max_input_length - len(input_ids)), 
                            attention_mask + [0] * (args.max_input_length - len(input_ids)), 
                            [-100] * (len(input_ids) - len(middle_ids)) + middle_ids + [-100] * (args.max_input_length - len(input_ids))
    )
    if len(feature.input_ids) != len(feature.target_ids) or len(feature.input_ids) != args.max_input_length:
        print(len(repo_content["input_ids"]))
        print(len(direct_content["input_ids"]))
        print(len(middle_ids))
    assert len(feature.input_ids) == len(feature.target_ids) == args.max_input_length
    return feature
  

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
        logger.info(f"{name} epoch {epoch}: {str(results)}")
    torch.save(generator.state_dict(), f"{args.output_dir}/result_{epoch}/model.pth")
    generator.train()
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid", action='store_true',
                        help="Whether to run validation.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run validation.")
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")  
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
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
    
    logger_path = os.path.join(args.output_dir, 'train.log') if args.do_train else os.path.join(args.output_dir, 'test')
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
    all_eval_examples = None
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)  
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
    
    if args.do_train:
        data = load_dataset(args.train_filename)
        train_data = data[:int(len(data)*0.9)]
        valid_data = data[int(len(data)*0.9):]
        train_features = []
        for example in tqdm(train_data):
            train_features.append(convert_example_to_feature(example.prefix, example.suffix, example.middle, example.relevant_code, tokenizer, args))
        train_dataset = MyDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps, 
                                      collate_fn=DoNothingCollator)
        
        # Prepare optimizer and schedule (linear warmup and decay) for generator
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        losses = []
        
        if args.do_valid:
            test({"github_project":valid_data}, generator, tokenizer, args, 0)
        
        for epoch in range(args.num_train_epochs):
            for batch in train_dataloader:
                generator.train()
                
                # Cat the ids of code, relevant document to form the final input
                inputs = [feature.input_ids for feature in batch]
                outputs = [feature.target_ids for feature in batch]
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                outputs = torch.tensor(outputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                results = generator(input_ids=inputs, 
                                    attention_mask=source_mask,
                                    labels=outputs)
                loss = results.loss
                
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
        
            if args.do_valid:
                test({"github_project":valid_data}, generator, tokenizer, args, epoch)
    
    if args.do_test:
        
        if all_eval_examples is None:
            # first time to do valid
            
            all_eval_examples = {
                "cceval_python": load_dataset("cceval_python"),
                "repoeval_line": load_dataset("repoeval_line")
            }
        
        
        test(all_eval_examples, generator, tokenizer, args, 0 if not args.do_train else epoch)
                
if __name__ == '__main__':
    main()  
