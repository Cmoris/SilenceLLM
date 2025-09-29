import json
import pandas as pd
from collections import defaultdict

# 读取CSV数据
csv_data = pd.read_csv("/n/work1/muyun/Dataset/datasets_v3/annotation.csv")  # 替换为你的CSV路径

# 构建标签映射：{date_id: {index: {"Stopped": 1, "Thinking": 0}}}
label_mapping = defaultdict(dict)
for _, row in csv_data.iterrows():
    file_name = row['file_name']
    stopped = row['Stopped']
    thinking = row['Thinking']
    
    parts = file_name.split('_')
    date_id = '_'.join(parts[:-1])  # 如 "20231215_03"
    index = int(parts[-1])          # 如 4
    
    label_mapping[date_id][index] = {
        "Stopped": stopped,
        "Thinking": thinking
    }

# 读取JSON数据
json_lines = []
with open("/misc/home/muyun/VScode/project/LLM/Silence/data/utils/silence_segments.jsonl", 'r') as f:  # 替换为你的JSON路径
    for line in f:
        json_lines.append(json.loads(line))

# 处理每个JSON对象，添加标签
new_json_lines = []
for entry in json_lines:
    filename = entry["filename"]
    date_id = filename.replace('_audio_subject.wav', '')
    
    segments = [seg for seg in entry["silence_segments"] if seg['end'] - seg['start'] > 2]
    new_segments = []
    
    # 为每个segment添加标签
    for i, segment in enumerate(segments):
        # 获取对应的标签（如果存在）
        if date_id in label_mapping and (i+1) in label_mapping[date_id]: 
            labels = label_mapping.get(date_id).get(i+1)
        else:
            continue
        
        # 创建新的segment对象，包含原有信息和标签
        new_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "Stopped": labels["Stopped"],
            "Thinking": labels["Thinking"]
        }
        new_segments.append(new_segment)
    
    # 更新entry
    entry["silence_segments"] = new_segments
    entry["filename"] = date_id
    new_json_lines.append(entry)

# 输出结果
for entry in new_json_lines:
    print(json.dumps(entry))
    
output_path = "/misc/home/muyun/VScode/project/LLM/Silence/data/utils/labels.jsonl"
with open(output_path, 'w+') as f:
    for entry in new_json_lines:
        json_data = json.dumps(entry)
        print(json_data)
        f.write(json_data)
        f.write("\n")
    