import argparse
import torch
from tqdm import tqdm
from model import build_model
from utils.util import (InputFeatures, Benchmarks, Example, CodeBlock)
from utils.eval_repoeval import compute_EM

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--generator_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--retriever_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base" ) 
    parser.add_argument("--weighted_parameters", default=None, type=str,
                        help="Path to .pth file: e.g. roberta-base" )   
    parser.add_argument('--sampled_code_num', default=10, type=int,
                        help="Total number of code blocks will be sampled")
    parser.add_argument('--max_input_length', default=2048, type=int,
                        help="Max token num for input feature")
    parser.add_argument('--max_crossfile_length', default=1536, type=int,
                        help="Max token num for crossfile")
    parser.add_argument('--max_generation_length', default=64, type=int,
                        help="Max token num for generating when evaluate")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    generator, _ = build_model(args)
    testsets_name = [
        "ours", 
        "ours_suffix", 
        "cceval_python", 
        "repoeval_line"
    ]
    benchmark = Benchmarks(generator.tokenizer, args)
    
    all_eval_features = {name:benchmark[name] for name in testsets_name}
    
    for name, features in all_eval_features.items():
        total = 0
        no_valid = 0
        with torch.no_grad():
            for i, feature in enumerate(features):
                decoder_features = [
                    InputFeatures(
                        feature.prefix_ids,
                        feature.suffix_ids,
                        feature.middle_ids,
                        document=[d]
                    ) for d in feature.document
                ]
                
                target = benchmark.test_datasets[name][i].middle
                generations = generator.generate(decoder_features, is_training=False)
                loss_batch = generator.generate(decoder_features, is_training=True)
                results = [compute_EM(target, [generation], 1) for generation in generations]
                
                if any(results):
                    total += 1
                    min = 10000000
                    max = 0
                    for result, loss in zip(results, loss_batch):
                        if result and loss > max:
                            max = loss
                        elif not result and loss < min:
                            min = loss
                        
                    if max > min:
                        no_valid += 1
                        
                print(f"{name} : {no_valid} - {total}") 
                
                
if __name__ == "__main__":
    main()