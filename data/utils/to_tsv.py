import pandas as pd
import numpy as np
import json
from dataset import AVProcessingDataset

def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def get_data(id_, video_data, audio_data, text_data):
    N = len(video_data)
    data = []
    label = []
    for i in range(N):
        video_path = video_data[i]['path']
        audio_path = audio_data[i]['path']

        video_frames = video_data[i]['num_frames']
        audio_frames = audio_data[i]['num_frames']

        text = text_data[i]['text'].split()
        num_words = len(text)
        speech_rates = audio_frames / num_words
        text = ' '.join(text)

        data.append([id_, video_path, audio_path, text, video_frames, audio_frames, speech_rates])
        label.append(video_data[i]['label'])

    return data, label

def save(data, label, split):
    df = pd.DataFrame(data)
    df.to_csv(f"/misc/home/muyun/VScode/project/LLM/MMS-LLaMA-gate/manifest/{split}.tsv", sep='\t', index=False)
    with open(f"/misc/home/muyun/VScode/project/LLM/MMS-LLaMA-gate/manifest/{split}.wrd", "wb") as f:
        content = "\n".join(label).encode("utf-8")
        f.write(content)

audio_dir = "/n/work1/muyun/Dataset/datasets/custom_sft/audio"
video_dir = "/n/work1/muyun/Dataset/datasets/custom_sft/video"

audio_train_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_a.json')
video_train_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_v.json')
text_train_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_t.json')

audio_test_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_a_test.json')
video_test_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_v_test.json')
text_test_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_t_test.json')

audio_val_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_a_val.json')
video_val_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_v_val.json')
text_val_data = AVProcessingDataset('/n/work1/muyun/Dataset/datasets/custom_sft/config_t_val.json')

train_data, train_label = get_data(video_train_data, audio_train_data, text_train_data)
test_data, test_label = get_data(video_test_data, audio_test_data, text_test_data)
val_data, val_label = get_data(video_val_data, audio_val_data, text_val_data)

save(train_data, train_label, 'train')
save(test_data, test_label, 'test')
save(val_data, val_label, 'val')
