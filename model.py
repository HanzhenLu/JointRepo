# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Tuple

        
class DataBase(object):
    def __init__(self, vector) -> None:
        self.vector = vector
        self.history = []
        
    def __len__(self):
        return self.vector.shape[0]
    
    def search(self, query, number, stage=None):
        scores = np.matmul(query, self.vector.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
        
        index = []
        for i in range(len(sort_ids)):
            if stage == "train":
                index.append(sort_ids[i][1:number+1])
            else:
                index.append(sort_ids[i][0:number])
        
        if stage == "train":
            self.history.append(index)
                
        return index
    
    def get_history(self):
        temp = self.history
        self.history = []
        return temp
    
    def update(self, index, vectors):
        for id, vector in zip(index, vectors):
            self.vector[id] = vector

logger = logging.getLogger(__name__)

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.sep_token:
        tokenizer.add_special_tokens(
            {'sep_token': '<SEP>'}
        )
        logger.info("Add <SEP> into the tokenizer")
    generator = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    logger.info("Finish loading generator [%s] from %s", get_model_size(generator), args.model_name_or_path)

    return generator, tokenizer