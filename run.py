from __future__ import absolute_import
import os
import torch
import random
import logging
import argparse
import json
import numpy as np
from colorama import Fore
from tqdm import tqdm
from fastbm25 import fastbm25
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer)
from transformers.generation.stopping_criteria import StoppingCriteriaList

from util import (load_train_and_valid_dataset, load_test_dataset,
                  label_line, CodeBlock, split_into_smaller_blocks, split_word, 
                  get_NL_list, StopAtSpecificTokenCriteria)
from eval import evaluate
from model import (build_model)

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
        
def convert_example_to_feature(prefix:str, suffix:str, middle:str, related_files:List[CodeBlock], tokenizer:PreTrainedTokenizer, do_generate:bool, args) -> InputFeatures:
    prefix_id = tokenizer.convert_tokens_to_ids("<PREFIX>")
    suffix_id = tokenizer.convert_tokens_to_ids("<SUFFIX>")
    middle_id = tokenizer.convert_tokens_to_ids("<MIDDLE>")
    
    code_blocks:List[CodeBlock] = []
    for file in related_files:
        code_blocks.extend(split_into_smaller_blocks(file, args.enable_fixed_block))
    
    corpus = [x.code_content for x in code_blocks]
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
    # tokenized_corpus = []
    # for code in corpus:
    #     tokenized_corpus.append(tokenizer.tokenize(code))
    # for code in corpus:
    #     code_token = code.split()
    #     tokens = []
    #     for token in code_token:
    #         tokens.extend(split_word(token))
    #     tokenized_corpus.append(" ".join(tokens))
    # if len(tokenized_corpus) == 0:
    #     print([str(block) for block in related_files])
    #     exit()
    bm25_model = fastbm25(tokenized_corpus)
    prefix_line = prefix.split("\n")
    # TODO: set the query_line as a hyperparameter
    if len(prefix_line) >= 15:
        query = "\n".join(prefix_line[-15:])
    else:
        query = "\n".join(prefix_line)
        suffix_line = suffix.split("\n")
        query += "\n" + "\n".join(suffix_line[:15-len(prefix_line)])
    query = tokenizer.tokenize(query)
    # query_token = prefix.split()
    # tokens = []
    # for token in query_token:
    #     tokens.extend(split_word(token))
    result = bm25_model.top_k_sentence(query, k=args.relevant_code_num)
    related_codes = [corpus[x[1]] for x in result if x[2] > 3]
    # print(related_codes)
    if len(related_codes) > 0:
        related_tokenized_result = tokenizer(related_codes, add_special_tokens=False)
        
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    middle_tokenized_result = tokenizer(middle, add_special_tokens=False)
    prefix_length = int(0.75 * args.max_input_length / 2)
    suffix_length = int(0.75 * args.max_input_length - prefix_length)
    prefix_ids = prefix_tokenized_result["input_ids"] if len(prefix_tokenized_result["input_ids"]) < prefix_length else prefix_tokenized_result["input_ids"][-prefix_length:]
    suffix_ids = suffix_tokenized_result["input_ids"] if len(suffix_tokenized_result["input_ids"]) < suffix_length else suffix_tokenized_result["input_ids"][:suffix_length]
    middle_ids = middle_tokenized_result["input_ids"]
    direct_content = {
        "input_ids": [suffix_id] + suffix_ids + [prefix_id] + prefix_ids + [middle_id] + middle_ids,
        "attention_mask": [1] * (len(prefix_ids) + len(suffix_ids) + len(middle_ids) + 3)
    }
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    left_budget = args.max_input_length - len(direct_content["input_ids"])
    related_idx = 0
    while related_idx < len(related_codes) and len(repo_content["input_ids"]) + len(related_tokenized_result["input_ids"][related_idx]) < left_budget:
        repo_content["input_ids"].extend(related_tokenized_result["input_ids"][related_idx] + [tokenizer.sep_token_id])
        repo_content["attention_mask"].extend(related_tokenized_result["attention_mask"][related_idx] + [1])
        related_idx += 1
    
    input_ids = repo_content["input_ids"] + direct_content["input_ids"]
    attention_mask = repo_content["attention_mask"] + direct_content["attention_mask"]
    # input_ids = [] + direct_content["input_ids"]
    # attention_mask = [] + direct_content["attention_mask"]
    if do_generate:
        feature = InputFeatures(input_ids, attention_mask, None)
    else:
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

# def convert_examples_to_features(examples:List[List[Tuple[str, str]]], tokenizer:PreTrainedTokenizer, args) -> List[InputFeatures]:
#     features = []
    
#     for example in tqdm(examples):
#         items = construct_data(example)
#         if items is None:
#             continue
#         else:
#             prefix, suffix, middle, related_files = items

#         middle_tokenized_result = tokenizer(middle, add_special_tokens=False)
#         if len(middle_tokenized_result["input_ids"]) > 50:
#             continue
#         feature = convert_example_to_feature(prefix, suffix, middle, related_files, tokenizer, False, args)
        
#         features.append(feature)
#     return features
  

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
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)  
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
    
    if args.do_train:
        # Prepare training data loader
        # train_examples, valid_examples = load_train_and_valid_dataset(args.train_filename)
        # train_features = convert_examples_to_features(train_examples, tokenizer)
        data = load_test_dataset(args, args.train_filename)
        train_data = data[:int(len(data)*0.9)]
        valid_data = data[int(len(data)*0.9):]
        train_features = []
        for items in tqdm(train_data):
            train_features.append(convert_example_to_feature(items[0], items[1], items[2], items[3], tokenizer, False, args))
        valid_features = []
        for items in tqdm(valid_data):
            valid_features.append(convert_example_to_feature(items[0], items[1], items[2], items[3], tokenizer, False, args))
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
        all_eval_examples = None
        
        for epoch in range(args.num_train_epochs):
            # for batch in train_dataloader:
            #     generator.train()
                
            #     # Cat the ids of code, relevant document to form the final input
            #     inputs = [feature.input_ids for feature in batch]
            #     outputs = [feature.target_ids for feature in batch]
            #     inputs = torch.tensor(inputs, dtype=torch.long).to(device)
            #     outputs = torch.tensor(outputs, dtype=torch.long).to(device)
            #     source_mask = inputs.ne(tokenizer.pad_token_id)
            #     results = generator(input_ids=inputs, 
            #                         attention_mask=source_mask,
            #                         labels=outputs)
            #     loss = results.loss
                
            #     if args.n_gpu > 1:
            #         loss = loss.mean()
            #     if args.gradient_accumulation_steps > 1:
            #         loss = loss / args.gradient_accumulation_steps
            #     losses.append(loss.item())
            #     loss.backward()
            #     if len(losses) % args.gradient_accumulation_steps == 0:
            #         # Update parameters
            #         optimizer.step()
            #         optimizer.zero_grad()
            #         scheduler.step()
            #         if len(losses) // args.gradient_accumulation_steps % 100 == 0:
            #             logger.info("epoch {} step {} loss {}".format(epoch,
            #                                          len(losses)//args.gradient_accumulation_steps,
            #                                          round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
        
            if args.do_valid:
                
                if all_eval_examples is None:
                    # first time to do valid
                    
                    all_eval_examples = {
                        "github_eval": valid_data,
                        "cceval_python": load_test_dataset(args, "data/cceval/python/test.parquet"),
                    }
                    
                    stopping_criteria = StoppingCriteriaList()
                    NL_list = get_NL_list(tokenizer)
                    stopping_criteria.append(StopAtSpecificTokenCriteria(NL_list))
                
                for name, examples in all_eval_examples.items():
                    print("Evaluating on {} dataset".format(name))
                    references, predictions = [], []
                    with torch.no_grad():
                        generator.eval()
                        for example in tqdm(examples):
                            # if name == "cceval_python":
                            prefix = example[0]
                            suffix = example[1]
                            references.append(example[2])
                            related_files = example[3]
                            # elif name == "github_eval":
                            #     items = construct_data(example)
                            #     if items is None:
                            #         continue
                            #     else:
                            #         prefix, suffix, middle, related_files = items
                            #         references.append(middle)
                            
                            output_text = '\n'
                            generated_code = ''
                            stack = []
                            max_try = 3
                            # 当模型只输出空白行，或者输出内容没能闭包时，继续输出
                            while (output_text == '\n' or len(stack) > 0) and max_try > 0:
                                max_try -= 1
                                prefix = prefix + generated_code
                                feature:InputFeatures = convert_example_to_feature(prefix, suffix, "", related_files, tokenizer, True, args)
                                generated_ids = generator.generate(input_ids=torch.tensor([feature.input_ids]).to(generator.device), attention_mask=torch.tensor([feature.attention_mask]).to(generator.device), max_new_tokens=50, do_sample=False, \
                                                                stopping_criteria=stopping_criteria, pad_token_id=tokenizer.eos_token_id)
                                output_text = tokenizer.decode(generated_ids[0][len(feature.input_ids):], skip_special_tokens=True)
                                for c in output_text:
                                    if c == '{' or c == '[' or c == '(':
                                        stack.append(c)
                                    elif c == '}' or c == ']' or c== ')':
                                        # 有可能 {, [, ( 来自prefix
                                        if len(stack) > 0:
                                            last_c = stack.pop()
                                            match_c = last_c + c
                                            if match_c == "{}" or match_c == "[]" or match_c == "()":
                                                continue
                                            else:
                                                stack = []
                                                break
                                generated_code += output_text
                            
                            
                            
                            predictions.append(generated_code)
                        generator.train()
                    
                    edit_similarity, exact_match = evaluate(references, predictions)
                    print("{} dataset: edit_similarity: {}%, exact_match: {}%".format(name, edit_similarity*100, exact_match*100))
                    with open(os.path.join(args.output_dir, f"{epoch}-{name}.jsonl"), 'w') as f:
                        for ref, pre in zip(references, predictions):
                            js = {
                                "reference":ref,
                                "prediction":pre
                            }
                            f.write(json.dumps(js)+'\n')
                
if __name__ == '__main__':
    main()  
