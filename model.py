# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 
import numpy as np
import logging
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Tuple, List, Union

from utils.util import (get_cross_file_context,
                        CodeBlock, InputFeatures)

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if "snowflake" in model_name_or_path.lower():
            self.model = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=False, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tempurate = 1.0
    
    def embedding(self, input_str:Union[str, List[str]], is_query:bool) -> torch.Tensor:
        if is_query:
            self.tokenizer.truncation_side = "left"
        else:
            self.tokenizer.truncation_side = "right"
        input_ids = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        embedding = self.model(**input_ids)[0][:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def gumbel_retrieve(self, query_str:str, documents:List[CodeBlock], n:int=None) \
        -> Union[torch.Tensor, List[CodeBlock]]:
        assert n is not None
            
        if len(documents) == 0:
            return []
        
        # 提供的文档数可能小于需要的文档数
        n = min(n, len(documents))
        
        self.eval()
        with torch.no_grad():
            query_embedding = self.embedding(query_str, True)
            document_embedding = self.embedding([doc.code_content for doc in documents], False)
            scores = torch.matmul(query_embedding, document_embedding.T).squeeze(dim=0)
            scores = torch.nn.functional.gumbel_softmax(scores, tau=self.tempurate)
        self.train()
        
        _, top_indices = torch.topk(scores, n)
        top_indices = top_indices.cpu().tolist()
        return [documents[i] for i in top_indices]
    
    def retrieve(self, query_str:str, documents:List[CodeBlock], n:int=None, is_training:bool=True) \
        -> Union[torch.Tensor, List[CodeBlock]]:
        r'''
        Params:
            query_str:用于检索的字符串
            document_str:被检索的文档合集
            is_training:训练时返回各个document的分数；非训练时返回分数最高的n个document
        '''
        
        if not is_training:
            assert n is not None
            
            if len(documents) == 0:
                return []
            
            # 提供的文档数可能小于需要的文档数
            n = min(n, len(documents))
            
            self.eval()
            with torch.no_grad():
                query_embedding = self.embedding(query_str, True)
                document_embedding = self.embedding([doc.code_content for doc in documents], False)
                scores = torch.matmul(query_embedding, document_embedding.T).squeeze(dim=0)
            self.train()
            
            _, top_indices = torch.topk(scores, n)
            top_indices = top_indices.cpu().tolist()
            return [documents[i] for i in top_indices]
        
        else:
            assert len(documents) != 0
            
            query_embedding = self.embedding(query_str, True)
            document_embedding = self.embedding([doc.code_content for doc in documents], False)
            scores = torch.matmul(query_embedding, document_embedding.T).squeeze(dim=0)
        
            return scores
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
class UnixcoderForRetriever(Retriever):
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tempurate = 1.0
        
    def tokenize(self, input_str, is_query):
        tokens = self.tokenizer.tokenize(input_str)
        max_length = 512
        if is_query:
            tokens = tokens[-(max_length - 4):]
        else:
            tokens = tokens[:(max_length - 4)]
        tokens = [self.tokenizer.cls_token, "<encoder-only>", self.tokenizer.sep_token] \
            + tokens + [self.tokenizer.sep_token]
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_length - len(tokens_id)
        tokens_id += [self.tokenizer.pad_token_id] * padding_length
        return tokens_id
        
    def embedding(self, input_str, is_query):
        if not isinstance(input_str, str):
            source_ids = [self.tokenize(string, is_query) for string in input_str]
        else:
            source_ids = [self.tokenize(input_str, is_query)]
        
        source_ids = torch.tensor(source_ids, dtype=torch.long).to("cuda")
        mask = source_ids.ne(self.tokenizer.pad_token_id).to("cuda")
        token_embeddings = self.model(source_ids, attention_mask=mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
class Generator:
    def __init__(self, model_name_or_path:str, args):
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # 特殊标记ID映射
        # TODO: 这需要根据模型进行改动
        if "opc" in model_name_or_path or "llama" in model_name_or_path or "pretrain" in model_name_or_path:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<PREFIX>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<SUFFIX>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<MIDDLE>"),
                "eos_id": self.tokenizer.convert_tokens_to_ids("<EOS>")
            }
        elif "deepseek" in model_name_or_path:
            self.special_token_ids = {
                "prefix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>"),
                "suffix_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁end｜>"),
                "middle_id": self.tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>"),
                "eos_id": self.tokenizer.eos_token_id
            }
        else:
            raise RuntimeError("Unknown generator")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def truncate_context(self, prefix_ids, suffix_ids, allocated_length):
        # 根据allocated_length尽量均衡地截断上文和下文
        
        prefix_length = allocated_length // 2
        suffix_length = allocated_length - prefix_length
        if len(prefix_ids) < prefix_length and len(suffix_ids) < suffix_length:
            return prefix_ids, suffix_ids
            
        elif len(prefix_ids) < prefix_length:
            suffix_length = allocated_length - len(prefix_ids)
            return prefix_ids, suffix_ids[:suffix_length]
        
        elif len(suffix_ids) < suffix_length:
            prefix_length = allocated_length - len(suffix_ids)
            return prefix_ids[-prefix_length:], suffix_ids
        else:
            return prefix_ids[-prefix_length:], suffix_ids[:suffix_length]
        
    def construct_prompt(self, prefix_ids, suffix_ids, cross_file_context, middle_ids=None):
        if "opc" in self.args.generator_name_or_path or "llama" in self.args.generator_name_or_path or "pretrain" in self.args.generator_name_or_path:
            input_ids = [self.special_token_ids["suffix_id"]] + suffix_ids + [self.special_token_ids["prefix_id"]] \
                + cross_file_context["input_ids"] + prefix_ids + [self.special_token_ids["middle_id"]]
            
        elif "deepseek" in self.args.generator_name_or_path:
            input_ids = [32013] + [self.special_token_ids["prefix_id"]] + cross_file_context["input_ids"] \
                + prefix_ids + [self.special_token_ids["middle_id"]] + suffix_ids + [self.special_token_ids["suffix_id"]]
        
        if middle_ids is not None:
            input_ids = input_ids + middle_ids + [self.special_token_ids["eos_id"]]
        
        return input_ids
    
    def generate(self, input_batch:List[InputFeatures], is_training:bool):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in input_batch:
            # 处理跨文件上下文
            cross_file_context = get_cross_file_context(feature.document, self.tokenizer, self.args.max_crossfile_length)
            
            middle_ids = feature.middle_ids[:self.args.max_generation_length]
            
            # 计算分配长度
            if is_training:
                max_allocated_length = self.args.max_input_length - self.args.max_crossfile_length - len(middle_ids) - 5
            else:
                max_allocated_length = self.args.max_input_length - len(cross_file_context["input_ids"]) - 4
            
            prefix_ids, suffix_ids = self.truncate_context(feature.prefix_ids, feature.suffix_ids, max_allocated_length)
            
            if is_training:
                input_ids = self.construct_prompt(prefix_ids, suffix_ids, cross_file_context, middle_ids)
                
                labels = [-100] * (len(input_ids) - len(middle_ids) - 1) \
                    + middle_ids + [self.special_token_ids["eos_id"]]
                attention_mask = [1] * len(input_ids)
                padding_length = self.args.max_input_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            else:
                input_ids = self.construct_prompt(prefix_ids, suffix_ids, cross_file_context)
                
                attention_mask = [1] * len(input_ids)
                padding_length = self.args.max_input_length - len(input_ids)
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
        attention_tensor = torch.tensor(batch_attention_mask, dtype=torch.bool).to(self.device)
        if is_training:
            label_tensor = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
            outputs = self.model(input_ids=input_tensor,
                                attention_mask=attention_tensor,
                                labels=label_tensor)
            
            batch_logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            
            shift_logits = batch_logits[..., :-1, :].contiguous()
            shift_labels = label_tensor[..., 1:].contiguous()
            
            per_position_loss = loss_fct(
                shift_logits.view(-1, batch_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_logits.size(0), -1)

            # 创建掩码，标记有效标签位置（不等于-100）
            mask = (shift_labels != -100)
            
            # 计算每个样本的总损失（忽略的位置损失已为0）
            per_sample_loss_sum = per_position_loss.sum(dim=1)
            
            # 计算每个样本的有效标签数量，并将0替换为1避免除以0
            valid_counts = mask.sum(dim=1).float()
            valid_counts[valid_counts == 0] = 1.0
            
            # 计算每个样本的平均损失
            per_sample_loss = per_sample_loss_sum / valid_counts
            return per_sample_loss
            
        else:
            batch_generated_ids = []
            pbar = tqdm(zip(input_tensor, attention_tensor), desc="Generating", total=len(input_tensor))
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

    def check_generation(self, input_batch:List[InputFeatures]) -> List[bool]:
        self.eval()
        output = []
        
        for feature in input_batch:
            cross_file_context = get_cross_file_context(feature.document, self.tokenizer, self.args.max_crossfile_length)
            middle_ids = feature.middle_ids[:self.args.max_generation_length]
            max_allocated_length = self.args.max_input_length - self.args.max_crossfile_length - len(middle_ids) - 5
            prefix_ids, suffix_ids = self.truncate_context(feature.prefix_ids, feature.suffix_ids, max_allocated_length)
            
            input_ids = self.construct_prompt(prefix_ids, suffix_ids, cross_file_context)
            labels = feature.middle_ids

            # 处理labels为空的情况
            n = len(labels)
            if n == 0:
                output.append(True)
        
            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            input_ids = input_ids
            m = input_ids.size(0)
        
            max_length = m + n - 1
            batch_size = n
        
            # 初始化批处理张量
            input_ids_batch = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            attention_mask_batch = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
        
            # 填充每个样本的输入和注意力掩码
            for i in range(n):
                # 填充原始输入
                input_ids_batch[i, :m] = input_ids
                
                # 填充标签部分（当i>0时）
                if i > 0:
                    input_ids_batch[i, m:m+i] = labels[:i]
                
                # 设置注意力掩码
                attention_mask_batch[i, :m+i] = 1
        
            # 模型前向传播
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        
            # 获取各样本最后一个有效位置的logits
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            lengths = m + torch.arange(batch_size, device=self.device)  # 各样本有效长度
            last_indices = lengths - 1  # 最后有效位置索引
        
            # 收集预测结果
            batch_indices = torch.arange(batch_size, device=self.device)
            predicted_token_ids = logits[batch_indices, last_indices].argmax(dim=-1).cpu()
        
            # 验证结果
            output.append(torch.all(predicted_token_ids == labels).item())
        self.train()
        return output
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
            
def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args) -> Tuple[Generator, Retriever]:
    generator = Generator(args.generator_name_or_path, args)
    if args.retriever_name_or_path is None:
        retriever = None
    elif "unixcoder" in args.retriever_name_or_path.lower():
        logger.info("Using child class UnixcoderForRetriever !!!")
        retriever = UnixcoderForRetriever(args.retriever_name_or_path)
    else:
        retriever = Retriever(args.retriever_name_or_path)
        
    if args.weighted_parameters is not None:
        logger.info(f"load parameters from {args.weighted_parameters}")
        generator.model.load_state_dict(torch.load(os.path.join(args.weighted_parameters, "generator.pth")))
        retriever.model.load_state_dict(torch.load(os.path.join(args.weighted_parameters, "retriever.pth")))
    
    generator.model.to(args.device)  
    logger.info("Finish loading generator [%s] from %s", get_model_size(generator.model), args.generator_name_or_path)
    
    if retriever is not None:
        retriever.model.to(args.device)
        logger.info("Finish loading retriever [%s] from %s", get_model_size(retriever.model), args.retriever_name_or_path)
    
    if args.n_gpu > 1:
        generator.model = torch.nn.DataParallel(generator.model)
        if retriever is not None:
            retriever.model = torch.nn.DataParallel(retriever.model)
    
    return generator, retriever