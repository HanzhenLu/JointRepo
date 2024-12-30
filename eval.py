import json
import torch.multiprocessing as mp


from tqdm import tqdm
from functools import partial
from typing import List, Tuple


from utils.eval_util import process_examples

def levenshtein_distance(str1, str2):
    # 创建一个二维数组来存储距离
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 初始化边界条件
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 计算编辑距离
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                           dp[i][j - 1] + 1,    # 插入
                           dp[i - 1][j - 1] + cost)  # 替换

    return dp[len1][len2]

def levenshtein_similarity(str1, str2):
    return 1 - (levenshtein_distance(str1, str2) / max(len(str1), len(str2)))

def evaluate(reference:List[str], prediction:List[str]) -> Tuple[float, float]:
    sum = 0
    correct = 0
    for ref, pre in zip(reference, prediction):
        sum += levenshtein_similarity(ref, pre)
        if ref.strip() == pre.strip():
            correct += 1
    return (sum / len(reference), correct / len(reference))

def compute_metric_stmt(output_dir, prompt_file, language="python"):
    task_ids = {}
    with open(f"{output_dir}/prediction.jsonl", "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            samples.append(json.loads(l))
            task_ids[json.loads(l)["task_id"]] = 1

    examples = {}
    with open(prompt_file, "r") as f_in:
        for l in f_in.readlines():
            ex = json.loads(l)
            if ex["metadata"]["task_id"] in task_ids:
                examples[ex["metadata"]["task_id"]] = {
                    "prompt": ex["prompt"],
                    "groundtruth": ex["groundtruth"]
                }
    
    assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"

    
    ts_lang = "c_sharp" if language == "csharp" else language

    truncated_samples = []
    em_labels = []

    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples, ts_lang)

    with tqdm(total=len(samples), disable=True) as pbar:
        for output in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            trunc_s, em_label = output
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    exact_match = 0
    with open(f"{output_dir}/prediction_truncated.jsonl", 'w', encoding="utf-8") as pt, \
            open(f"{output_dir}/exact_match_idx.jsonl", 'w') as em:
        for trunc_s, em_label in zip(truncated_samples, em_labels):
            pt.write(json.dumps(trunc_s) + "\n")
            if em_label == 1:
                em.write(f'{trunc_s["task_id"]}\n')
                exact_match += 1

    ### Score calculation

    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        es = levenshtein_similarity(trunc_s["target"], trunc_s["pred"])
        edit_similarities.append(es)

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em_labels[idx],
            "es": es,
        })

    em_ratio = round(exact_match / len(samples) * 100, 4)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 4)

    with open(f"{output_dir}/detailed_results.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # eval_results = eval_repoeval(f"{output_dir}/prediction_truncated.jsonl")
    # em_ratio = f"{em_ratio}({eval_results['em']})"
    # edit_sim = f"{edit_sim}({eval_results['es']})"

    # write the results to a file
    with open(f"{output_dir}/results.json", 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))
    return {
        "em": em_ratio,
        "es": edit_sim,
        "total": len(truncated_samples)
    }