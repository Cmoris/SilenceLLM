from scipy.stats import binomtest  # 若 scipy 1.9+ 警告，可用 binomtest
import numpy as np
import json

def read_result(file):
    with open(file) as f:
        json_data = json.load(f)
    if type(json_data) is list:
        json_data = json_data[0]
    targets = np.array([x=="""{\"class\": 1, \"content\": \"thinking\"}""" for x in json_data["ref"]], dtype=np.int8)  
    preds = np.array([x=="""{\"class\": 1, \"content\": \"thinking\"}""" for x in json_data["hypo"]], dtype=np.int8)
    return targets, preds

y_true, pred_A = read_result("/misc/home/muyun/VScode/project/LLM/Silence/result/llama/result-siglip2-beats-no_sr-no_q-concat/result.json")
_, pred_B = read_result("/home/muyun/VScode/project/LLM/Silence/result/llama/result-avhubert-no_sr-concat/result.json")

# 先计算四格表
both_correct = np.sum((pred_A == y_true) & (pred_B == y_true))
A_correct_B_wrong = np.sum((pred_A == y_true) & (pred_B != y_true))
A_wrong_B_correct = np.sum((pred_A != y_true) & (pred_B == y_true))

from sklearn.metrics import f1_score

obs_diff = 0.859 - 0.416


n_perm = 5000
rng = np.random.RandomState(0)
count = 0
n = len(y_true)

for i in range(n_perm):
    # 对每个样本，有 50% 概率交换两个模型的预测（相当于在零假设下交换）
    swap_mask = rng.rand(n) < 0.5
    a = pred_A.copy()
    b = pred_B.copy()
    a[swap_mask], b[swap_mask] = b[swap_mask], a[swap_mask]
    diff = f1_score(y_true, a, average='binary') - f1_score(y_true, b, average='binary')
    if abs(diff) >= abs(obs_diff):
        count += 1

p_value = (count + 1) / (n_perm + 1)
print(f"Observed diff={obs_diff:.4f}, permutation p-value={p_value:.4f}")
