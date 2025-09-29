import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report

def evaluate(preds, target):
    all_preds = preds
    all_labels = target
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    return metrics

def save_result(file):
    with open(file) as f:
        json_data = json.load(f)
    if type(json_data) is list:
        json_data = json_data[0]
    targets = json_data["ref"]  
    preds = json_data["hypo"]
    metrics = classification_report(targets, preds, digits=4)
    print(metrics)

path = "/misc/home/muyun/VScode/project/LLM/Silence/result/llama/result-siglip2-beats-no_sr-no_q-concat"
p = Path(path)
json_files = list(p.rglob('*.json'))

for json_file in json_files:

    print(json_file.parent.parent.name, json_file.parent.name)
    save_result(str(json_file))
    
    

