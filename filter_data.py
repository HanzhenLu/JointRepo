import pickle
import argparse
import random
import torch
from functools import partial
from tqdm import tqdm
from pathlib import Path
from model import build_model
from utils.util import convert_example_to_feature, InputFeatures, CodeBlock
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--retriever_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base" ) 
    parser.add_argument('--max_crossfile_length', default=1536, type=int,
                        help="Max token num for crossfile")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_generation_length', default=64, type=int,
                        help="Max token num for generating when evaluate")
    parser.add_argument('--GPU_ids', type=str, default='0',
                        help="The ids of GPUs will be used")
    
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_examples_num", type=int, default=100000)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    generator, _ = build_model(args)
    generator.eval()
    
    tokenizer_name = Path(args.generator_name_or_path).parts[-1]
    with open(f"preprocessed/{args.dataset_name}-{tokenizer_name}-{args.relevant_code_num}.pkl", 'rb') as f:
        examples = pickle.load(f)
    
    _examples = examples
    examples = [example for example in _examples if len(example.relevant_code) > 1]
    random.shuffle(examples)
    filter_examples = []
    
    # 多线程进行转换
    tokenizer = generator.tokenizer
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
    
    for idx, feature in tqdm(enumerate(features)):
        if len(feature.document) < 1:
            continue
        feature:InputFeatures
        decoder_features = [
            InputFeatures(
                feature.prefix_ids,
                feature.suffix_ids,
                feature.middle_ids,
                document=[documents]
            )
            for documents in feature.document[:10]
        ]
        # decoder_features.append(InputFeatures(
        #         feature.prefix_ids,
        #         feature.suffix_ids,
        #         feature.middle_ids,
        #         document=[CodeBlock("Unknown", "")]
        # ))
        
        with torch.no_grad():
            feature_loss = generator.generate(decoder_features, True)
            mean = feature_loss.mean().item()
            std = feature_loss.std().item()
            cv = std / mean
            if cv > 0.5:
                # assert examples[idx].relevant_code == feature.document
                filter_examples.append(examples[idx])
                if len(filter_examples) >= args.output_examples_num:
                    break
    
    with open(f"preprocessed/filted-{args.dataset_name}-{tokenizer_name}-{args.relevant_code_num}.pkl", 'wb') as f:
        pickle.dump(filter_examples, f)

if __name__ == "__main__":
    main()
