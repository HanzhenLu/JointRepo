from __future__ import absolute_import
import os
import torch
import logging
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import List
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)

from utils.util import (load_dataset, set_seed, convert_example_to_feature,
                        InputFeatures, Benchmarks, Example, CodeBlock)
from model import (build_model, Generator, Retriever)
from eval import compute_metric_stmt

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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

def test(benchmark:Benchmarks, names:List[str], generator:Generator, retriever:Retriever, args, epoch:int):
    generator.eval()
    retriever.eval()
    all_eval_examples = {name:benchmark[name] for name in names}
    for name, features in all_eval_examples.items():
        logger.info("Evaluating on {} dataset".format(name))
        decoder_features = []
        with torch.no_grad():
            for feature in features:
                documents = retriever.retrieve(feature.query, feature.document, 
                                               args.relevant_code_num, False)
                decoder_features.append(
                    InputFeatures(
                        feature.prefix_ids,
                        feature.suffix_ids,
                        feature.middle_ids,
                        document=documents
                    )
                )
            
            generations = generator.generate(decoder_features, is_training=False)
        if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
            os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
        with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
            for example, generation in zip(benchmark.test_datasets[name], generations):
                f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")
        if name == "cceval_python":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl")
        elif name == "repoeval_line":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl")
        elif name == "ours":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/ours/python/test.jsonl")
        elif name == "ours_suffix":
            results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/ours/python/test_suffix.jsonl")
        elif name == "validation" or name == "train":
            targets, generations = ["".join(x.middle.split()) for x in benchmark.test_datasets[name]], ["".join(x.split()) for x in generations]
            results = {}
            results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
        else:
            raise Exception("unsupport test set")
        logger.info(f"{name} epoch {epoch}: {str(results)}")
    if hasattr(generator.model, "module"):
        generator_to_save = generator.model.module
        retriever_to_save = retriever.model.module
    else:
        generator_to_save = generator.model
        retriever_to_save = retriever.model
    torch.save(generator_to_save.state_dict(), f"{args.output_dir}/result_{epoch}/generator.pth")
    torch.save(retriever_to_save.state_dict(), f"{args.output_dir}/result_{epoch}/retriever.pth")
    
    generator.train()
    retriever.train()

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--generator_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--retriever_name_or_path", default=None, type=str, required=False,
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
    parser.add_argument("--debug", action='store_true',
                        help="Whether to spped up training by reducing the size of training set")
    
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
    
    parser.add_argument('--relevant_code_num', default=1, type=int,
                        help="Total number of relevant code blocks to use")
    parser.add_argument('--sampled_code_num', default=10, type=int,
                        help="Total number of code blocks will be sampled")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_crossfile_length', default=1536, type=int,
                        help="Max token num for crossfile")
    parser.add_argument('--max_generation_length', default=64, type=int,
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
    generator, retriever = build_model(args)
    testsets_name = [
        "ours", 
        "ours_suffix", 
        "cceval_python", 
        "repoeval_line"
    ]
    
    tokenizer_name = Path(args.generator_name_or_path).parts[-1]
    benchmark = Benchmarks(generator.tokenizer, args)
    
    logger.info("Training/evaluation parameters %s", args)
    
    if args.do_train:
        data = load_dataset(args.train_filename, tokenizer_name, args.sampled_code_num * 5)
        if args.debug:
            random.shuffle(data)
            train_data = data[:8000]
            valid_data = data[-800:]
            benchmark.test_datasets["train"] = train_data[:len(train_data)//10]
        else:
            train_data = data[:int(len(data)*0.9)]
            # train_data = data
            valid_data = data[int(len(data)*0.9):]
            benchmark.test_datasets["train"] = data[int(len(data)*0.8):int(len(data)*0.9)]
        benchmark.test_datasets["validation"] = valid_data
        train_features = []
        for example in tqdm(train_data, desc="convert examples to features"):
            # 这种数据没有学习意义
            if len(example.relevant_code) <= 1:
                continue
            train_feature = convert_example_to_feature(example.prefix, example.suffix, example.middle, \
                example.relevant_code, generator.tokenizer)
            train_features.append(train_feature)
        
        train_dataset = MyDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps, 
                                      collate_fn=DoNothingCollator)
        
        # Prepare optimizer and schedule (linear warmup and decay) for generator
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in generator.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in retriever.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in retriever.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
        
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        losses = []
        softmax = torch.nn.Softmax(dim=-1)
        
        for epoch in range(1, args.num_train_epochs+1):
            for batch in train_dataloader:
                generator.train()
                retriever.train()
                
                loss = None
                for feature in batch:
                    feature:InputFeatures
                    relevant_documents = retriever.retrieve(feature.query, feature.document, args.sampled_code_num, False)
                    scores = retriever.retrieve(feature.query, relevant_documents)
                    decoder_features = [
                        InputFeatures(
                            feature.prefix_ids,
                            feature.suffix_ids,
                            feature.middle_ids,
                            document=[document]
                        )
                        for document in relevant_documents
                    ]
                    
                    feature_loss = generator.generate(decoder_features, True)

                    modified_score = softmax(scores)
                    current_loss = torch.dot(modified_score, feature_loss)
                    loss = current_loss if loss is None else loss + current_loss
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    clip_grad_norm_(
                        retriever.model.parameters(),
                        args.max_grad_norm
                    )
                    clip_grad_norm_(
                        generator.model.parameters(),
                        args.max_grad_norm
                    )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info(f"{feature_loss}")
                        logger.info(f"{scores}")
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
        
            if args.do_valid:
                test(benchmark, ["train", "validation"] + testsets_name, generator, retriever, args, epoch)
    
    if args.do_test and not args.do_valid:
        test(benchmark, testsets_name, generator, retriever, args, 0 if not args.do_train else epoch)
                
if __name__ == '__main__':
    main()  
