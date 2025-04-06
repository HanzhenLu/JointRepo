import pickle
import argparse
import random
import torch
import datasets
from functools import partial
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from model import build_model
from utils.util import convert_example_to_feature, InputFeatures, get_cross_file_context
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument('--max_crossfile_length', default=1536, type=int,
                        help="Max token num for crossfile")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_generation_length', default=50, type=int,
                        help="Max token num for generating when evaluate")
    parser.add_argument('--GPU_ids', type=str, default='0',
                        help="The ids of GPUs will be used")
    
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    
    special_token_ids = {
        "prefix_id": tokenizer.convert_tokens_to_ids("<PREFIX>"),
        "suffix_id": tokenizer.convert_tokens_to_ids("<SUFFIX>"),
        "middle_id": tokenizer.convert_tokens_to_ids("<MIDDLE>"),
        "eos_id": tokenizer.convert_tokens_to_ids("<EOS>"),
        "retrieval_start_id": tokenizer.convert_tokens_to_ids("<RETRIEVAL_START>"),
        "retrieval_end_id": tokenizer.convert_tokens_to_ids("<RETRIEVAL_END>")
    }
    
    with open(f"preprocessed/filter-opc-sft-v1-5.pkl", 'rb') as f:
        examples = pickle.load(f)
    
    _examples = examples
    examples = [example for example in _examples if len(example.relevant_code) >= 1]
    random.shuffle(examples)
    
    # 多线程进行转换
    convert_func = partial(convert_example_to_feature, tokenizer=tokenizer, args=args)
    with ThreadPoolExecutor(max_workers=16) as executor:
        task_args = (
            (ex.prefix, ex.suffix, ex.middle, ex.relevant_code)
            for ex in examples
        )
        features = list(tqdm(
            executor.map(
                lambda args: convert_func(*args),
                task_args
            ),
            total=len(examples),
            desc="Converting examples to features"
        ))
    
    assert len(features) == len(examples)
    
    output_dataset = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    for _, feature in tqdm(enumerate(features)):
        feature:InputFeatures
        document = [feature.document[0]] if len(feature.document) >= 1 else []
        cross_file_context = get_cross_file_context(document, tokenizer, args.max_crossfile_length)
        max_allocated_length = args.max_input_length - len(cross_file_context["input_ids"]) - len(feature.middle_ids) - 5
        prefix_length = max_allocated_length // 2
        suffix_length = max_allocated_length - prefix_length
        if len(feature.prefix_ids) < prefix_length and len(feature.suffix_ids) < suffix_length:
            prefix_ids = feature.prefix_ids
            suffix_ids = feature.suffix_ids
        elif len(feature.prefix_ids) < prefix_length:
            prefix_ids = feature.prefix_ids
            suffix_length = max_allocated_length - len(prefix_ids)
            suffix_ids = feature.suffix_ids[:suffix_length]
        elif len(feature.suffix_ids) < suffix_length:
            suffix_ids = feature.suffix_ids
            prefix_length = max_allocated_length - len(suffix_ids)
            prefix_ids = feature.prefix_ids[-prefix_length:]
        else:
            prefix_ids = feature.prefix_ids[-prefix_length:]
            suffix_ids = feature.suffix_ids[:suffix_length]
        
        input_ids = [special_token_ids["suffix_id"]] + suffix_ids + [special_token_ids["prefix_id"]] \
                    + [special_token_ids["retrieval_start_id"]] + cross_file_context["input_ids"] + [special_token_ids["retrieval_end_id"]] \
                    + prefix_ids + [special_token_ids["middle_id"]] + feature.middle_ids + [special_token_ids["eos_id"]]
        labels = [-100] * (len(input_ids) - len(feature.middle_ids) - 1) \
                    + feature.middle_ids + [special_token_ids["eos_id"]]
        attention_mask = [1] * len(input_ids)
        padding_length = args.max_input_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
        output_dataset["input_ids"].append(input_ids)
        output_dataset["attention_mask"].append(attention_mask)
        output_dataset["labels"].append(labels)
    
    output_dataset = datasets.Dataset.from_dict(output_dataset)
    output_dataset = output_dataset.train_test_split(test_size=0.0001)
    del(tokenizer)
    output_dataset_dict = datasets.DatasetDict({"train": output_dataset["train"], "validation": output_dataset["test"]})
    output_dataset_dict.save_to_disk("/data/hanzhenlu/dataset/repo-sft", num_proc=16)

if __name__ == "__main__":
    main()
