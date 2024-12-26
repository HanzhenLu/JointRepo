from typing import List, Tuple

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