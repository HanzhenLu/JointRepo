import argparse
import os
import json
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from tqdm import tqdm
from pathlib import Path
from utils.util import Example, CodeBlock, load_dataset, check_memory
from utils.eval_repoeval import compute_EM
from model import Retriever, UnixcoderForRetriever
from utils.eval_util import process_examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 prefix_ids:List[int],
                 suffix_ids:List[int],
                 middle_ids:List[int] = None,
                 query:str = None,
                 document:List[CodeBlock] = None,
                 project_name:str = None
    ):
        self.project_name = project_name
        self.prefix_ids = prefix_ids
        self.middle_ids = middle_ids
        self.suffix_ids = suffix_ids
        self.query = query
        self.document = document

def get_cross_file_context(related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int) -> Dict[str, List[int]]:
    filter_codeblocks = []
    
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        if file_path == "wrong_completion":
            filter_codeblocks.append("".join(["#" + line for line in code_content.splitlines(keepends=True)]))
        elif file_path != "" and file_path != "Unknown":
            filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    
    if len(filter_codeblocks) > 0:
        related_tokenized_result = tokenizer(filter_codeblocks, add_special_tokens=False)
    else:
        return repo_content
    
    for i in range(len(related_codes)):
        repo_content["input_ids"] += related_tokenized_result["input_ids"][i]
        repo_content["attention_mask"] += related_tokenized_result["attention_mask"][i]
    
    repo_content["input_ids"] = repo_content["input_ids"][:cross_file_budget]
    repo_content["attention_mask"] = repo_content["attention_mask"][:cross_file_budget]
    
    return repo_content

class Generator:
    def __init__(self, model_name_or_path:str, args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # 特殊标记ID映射
        # TODO: 这需要根据模型进行改动
        if "opc" in model_name_or_path or "llama" in model_name_or_path:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<PREFIX>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<SUFFIX>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<MIDDLE>"),
            }
        elif "deepseek" in model_name_or_path:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁end｜>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>"),
            }
        elif "Qwen" in model_name_or_path:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<|fim_prefix|>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<|fim_suffix|>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<|fim_middle|>"),
                "repo_name_id": self.tokenizer.convert_tokens_to_ids("<|repo_name|>"),
                "file_sep_id": self.tokenizer.convert_tokens_to_ids("<|file_sep|>")
            }
        else:
            raise RuntimeError("Unknown generator")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate(self, input_batch:List[InputFeatures], is_training:bool):
        batch_input_ids = []
        batch_attention_mask = []
        for feature in input_batch:
            # 处理跨文件上下文
            cross_file_context = get_cross_file_context(feature.document, self.tokenizer, self.args.max_crossfile_length)
            
            middle_ids = feature.middle_ids[:self.args.max_generation_length]
            
            # 计算分配长度
            if is_training:
                max_allocated_length = self.args.max_input_length - len(cross_file_context["input_ids"]) - len(middle_ids) - 5
            else:
                max_allocated_length = self.args.max_input_length - len(cross_file_context["input_ids"]) - 4
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
            
            if "opc" in self.args.generator_name_or_path or "llama" in self.args.generator_name_or_path:
                input_ids = [self.special_token_ids["suffix_id"]] + suffix_ids + [self.special_token_ids["prefix_id"]] \
                    + cross_file_context["input_ids"] + prefix_ids + [self.special_token_ids["middle_id"]]
            elif "deepseek" in self.args.generator_name_or_path:
                input_ids = [32013] + [self.special_token_ids["prefix_id"]] + cross_file_context["input_ids"] \
                    + prefix_ids + [self.special_token_ids["middle_id"]] + suffix_ids + [self.special_token_ids["suffix_id"]]
            elif "Qwen" in self.args.generator_name_or_path:
                input_ids = [self.special_token_ids["prefix_id"]] + prefix_ids + [self.special_token_ids["middle_id"]] + suffix_ids + [self.special_token_ids["suffix_id"]]
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_input_length - len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
            assert len(input_ids) == len(attention_mask)
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            
            
        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
        attention_tensor = torch.tensor(batch_attention_mask, dtype=torch.bool).to(self.device)
            
        batch_generated_ids = []
        pbar = tqdm(zip(input_tensor, attention_tensor), desc="Generating", total=len(input_tensor))
        with torch.no_grad():
            for input_ids, attention_mask in pbar:
                input_ids = input_ids.unsqueeze(dim=0)
                attention_mask = attention_mask.unsqueeze(dim=0)
                if hasattr(self.model, "module"):
                    model_to_generate = self.model.module
                else:
                    model_to_generate = self.model
                generated_ids = model_to_generate.generate(input_ids, attention_mask=attention_mask, 
                    max_length=input_ids.size(1)+self.args.max_generation_length, pad_token_id=self.tokenizer.pad_token_id)
                batch_generated_ids.extend(generated_ids[:, input_ids.size(1):])
        return [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in batch_generated_ids]
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--small_model_prediction", type=str, required=True)
    parser.add_argument("--generator_name_or_path", type=str, required=True)
    parser.add_argument("--retriever_name_or_path", type=str, required=True)
    parser.add_argument("--weighted_parameters", default=None, type=str,
                        help="Path to .pth file: e.g. roberta-base" )   
    parser.add_argument('--relevant_code_num', default=1, type=int,
                        help="Total number of relevant code blocks to use")
    parser.add_argument("--GPU_ids", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sampled_code_num", type=int, default=10)
    parser.add_argument("--using_completion", action="store_true")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_crossfile_length', default=1536, type=int,
                        help="Max token num for crossfile")
    parser.add_argument('--max_generation_length', default=64, type=int,
                        help="Max token num for generating when evaluate")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    if args.retriever_name_or_path is None:
        retriever = None
    elif "unixocder" in args.retriever_name_or_path.lower():
        retriever = UnixcoderForRetriever(args.retriever_name_or_path)
    else:
        retriever = Retriever(args.retriever_name_or_path)
    retriever.model.to(args.device)
    retriever.eval()
    generator = Generator(args.generator_name_or_path, args)
    generator.model.to(args.device)
    generator.eval()
    tokenizer_name = Path(args.generator_name_or_path).parts[-1]
    check_memory()
    
    test_datasets = {
        "ours": load_dataset("ours", tokenizer_name, args.sampled_code_num*5),
        "ours_suffix": load_dataset("ours-suffix", tokenizer_name, args.sampled_code_num*5),
        "cceval_python": load_dataset("cceval", tokenizer_name, args.sampled_code_num*5),
        "repoeval_line": load_dataset("repoeval", tokenizer_name, args.sampled_code_num*5)
    }
    
    for name, examples in test_datasets.items():
        passed_case = []
        wrong_completion = {}
        path = os.path.join(args.small_model_prediction, name, "prediction_truncated.jsonl")
        with open(path, 'r') as f:
            for line in f:
                js = json.loads(line)
                if compute_EM(js["target"], [js["pred"]], passk=1):
                    passed_case.append(js["task_id"])
                else:
                    wrong_completion[js["task_id"]] = js["pred"]
        
        features = []
        corresponding_examples = []
        for example in examples:
            if example.task_id in passed_case:
                continue
            prefix_line = example.prefix.split("\n")
            # 处理前缀部分：最多取8行
            prefix_part = prefix_line[-8:] if len(prefix_line) >= 8 else prefix_line
            remaining = 15 - len(prefix_part)  # 计算需要从后缀补充的行数

            # 处理后缀部分：过滤空行并取剩余需要的行数
            suffix_line_clean = [line for line in example.suffix.split('\n') if line.strip()]
            suffix_part = suffix_line_clean[:remaining]

            # 合并结果
            if args.using_completion:
                query_str = "\n".join(prefix_part) + wrong_completion[example.task_id] + "\n".join(suffix_part)
            else:
                query_str = "\n".join(prefix_part + suffix_part)
            
            # if args.using_completion:
            #     completion:str = wrong_completion[example.task_id]
            #     completion = "".join(["#" + line for line in completion.splitlines(keepends=True)])
            #     completion = "\n" + completion + "\n"
                
            #     completion_prefix = "\n".join(prefix_line[:-1]) + completion + prefix_line[-1]
            
            prefix_tokenized_result = generator.tokenizer(example.prefix, add_special_tokens=False)
            suffix_tokenized_result = generator.tokenizer(example.suffix, add_special_tokens=False)
            middle_tokenized_result = generator.tokenizer(example.middle, add_special_tokens=False)
            project_name = example.task_id.split("/")[0]
            project_name_result = generator.tokenizer(project_name, add_special_tokens=False)
            
            feature = InputFeatures(
                prefix_tokenized_result["input_ids"],
                suffix_tokenized_result["input_ids"],
                middle_tokenized_result["input_ids"],
                query_str,
                example.relevant_code,
                project_name_result["input_ids"]
            )
            features.append(feature)
            corresponding_examples.append(example)
        
        em_count = 0
        os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
        
        decoder_features, retrieved_codeblocks = [], []
        for feature in features:
            documents = retriever.retrieve(feature.query, feature.document, 
                                            args.relevant_code_num, False)
            retrieved_codeblocks.append(documents[0])
            
            decoder_features.append(
                InputFeatures(
                    feature.prefix_ids,
                    feature.suffix_ids,
                    feature.middle_ids,
                    document=documents
                )
            )
        
        generations = generator.generate(decoder_features, is_training=False)
        
        with open(os.path.join(args.output_dir, name, "prediction_truncated.jsonl"), 'w') as f:
            for generation, example, retrieved_codeblock in zip(generations, corresponding_examples, retrieved_codeblocks):
                trunc_s, em_label = process_examples("python", ({
                                                                    "task_id": example.task_id,
                                                                    "pred": generation
                                                                },
                                                                {
                                                                    "prompt": example.prefix, 
                                                                    "groundtruth": example.middle
                                                                }
                                                                ))
                em_count += em_label
                json_str = json.dumps({**trunc_s, "retrieved_codeblock":retrieved_codeblock.code_content})
                f.write(json_str+"\n")
        
        print(f"{name} : {em_count / len(test_datasets[name])}")
        

if __name__ == "__main__":
    main()