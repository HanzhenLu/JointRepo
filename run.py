from __future__ import absolute_import
import os
import torch
import random
import logging
import argparse
import json
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup, 
                          PreTrainedTokenizer, PreTrainedModel)
from utils.util import (load_dataset, cross_file_contexts, 
                        CodeBlock, Example, InputFeatures)
from pathlib import Path
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
        
def convert_example_to_feature(prefix:str, suffix:str, middle:str, related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, args) -> InputFeatures:
    prefix_id = tokenizer.convert_tokens_to_ids("<PREFIX>")
    suffix_id = tokenizer.convert_tokens_to_ids("<SUFFIX>")
    middle_id = tokenizer.convert_tokens_to_ids("<MIDDLE>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    
    # TODO: set the cross_file_budget as a hyperparameter
    repo_content = cross_file_contexts(related_codes, tokenizer, int(args.max_input_length * 0.75))
    
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    middle_tokenized_result = tokenizer(middle, add_special_tokens=False)
    
    left_budget = args.max_input_length - len(repo_content["input_ids"]) - len(middle_tokenized_result["input_ids"]) - 4
    prefix_length = int(left_budget / 2)
    suffix_length = int(left_budget - prefix_length)
    if len(prefix_tokenized_result["input_ids"]) < prefix_length and len(suffix_tokenized_result["input_ids"]) < suffix_length:
        prefix_ids = prefix_tokenized_result["input_ids"]
        suffix_ids = suffix_tokenized_result["input_ids"]
    elif len(prefix_tokenized_result["input_ids"]) < prefix_length:
        prefix_ids = prefix_tokenized_result["input_ids"]
        suffix_length = int(left_budget - len(prefix_ids))
        suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
    elif len(suffix_tokenized_result["input_ids"]) < suffix_length:
        suffix_ids = suffix_tokenized_result["input_ids"]
        prefix_length = int(left_budget - len(suffix_ids))
        prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
    else:
        prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
        suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
    middle_ids = middle_tokenized_result["input_ids"]

    input_ids = [suffix_id] + suffix_ids + [prefix_id] + repo_content["input_ids"] + prefix_ids + [middle_id] + middle_ids + [eos_id]
    attention_mask = [1] * len(input_ids)

    feature = InputFeatures(input_ids + [tokenizer.pad_token_id] * (args.max_input_length - len(input_ids)), 
                            attention_mask + [0] * (args.max_input_length - len(input_ids)), 
                            [-100] * (len(input_ids) - len(middle_ids) - 1) + middle_ids + [eos_id] + [-100] * (args.max_input_length - len(input_ids))
    )
    if len(feature.input_ids) != len(feature.target_ids) or len(feature.input_ids) != args.max_input_length:
        print(len(repo_content["input_ids"]))
        print(len(input_ids))
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
    parser.add_argument("--weighted_parameters", default=None, type=str,
                        help="Path to .pth file: e.g. roberta-base" )   
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
    
    logger_path = os.path.join(args.output_dir, 'train.log') if args.do_train else os.path.join(args.output_dir, 'test.log')
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
    tokenizer_name = Path(args.model_name_or_path).parts[-1]
    
    all_eval_examples = {
        "ours_suffix": load_dataset("ours-suffix", tokenizer_name, args),
        "ours": load_dataset("ours", tokenizer_name, args),
        "cceval_python": load_dataset("cceval", tokenizer_name, args),
        "repoeval_line": load_dataset("repoeval", tokenizer_name, args)
    }
    
    with open(f"{args.output_dir}/retrieval.pkl", 'wb') as f:
        pickle.dump(all_eval_examples, f)
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)  
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
    
    if args.do_train:
        data = load_dataset(args.train_filename, tokenizer_name, args)
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
        
        # if args.do_valid:
        #     test({**all_eval_examples, "github_projects":valid_data}, generator, tokenizer, args, 0)
        
        for epoch in range(1, args.num_train_epochs+1):
            for batch in train_dataloader:
                generator.train()
                
                # Cat the ids of code, relevant document to form the final input
                inputs = [feature.input_ids for feature in batch]
                attention_mask = [feature.attention_mask for feature in batch]
                outputs = [feature.target_ids for feature in batch]
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                attention_mask = torch.tensor(attention_mask, dtype=torch.bool).to(device)
                outputs = torch.tensor(outputs, dtype=torch.long).to(device)
                results = generator(input_ids=inputs, 
                                    attention_mask=attention_mask,
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
                test({**all_eval_examples, "github_projects":valid_data}, generator, tokenizer, args, epoch)
    
    if args.do_test and not args.do_valid:
        test(all_eval_examples, generator, tokenizer, args, 0 if not args.do_train else epoch)
                
if __name__ == '__main__':
    main()  
