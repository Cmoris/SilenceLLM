from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import subprocess

classes = [{"class": 0, "content": "stopped"}, {"class": 1, "content": "thinking"}]

def save_data(data, label, split):
    df = pd.DataFrame(data)
    df.to_csv(f"/n/work1/muyun/Dataset/datasets_v3/manifest/{split}.tsv", sep='\t', index=False)
    with open(f"/n/work1/muyun/Dataset/datasets_v3/manifest/{split}.wrd", "wb") as f:
        content = "\n".join(label).encode("utf-8")
        f.write(content)
        
    
def get_data(csv_path, video_dir, audio_dir):
    data = []
    df = pd.read_csv(csv_path)
    names = df.loc[:, "file_name"].values.tolist()
    labels = get_label(csv_path=csv_path)
    video_files = [Path(video_dir)/f"{name}.mp4" for name in names]
    audio_files = [Path(audio_dir)/f"{name}.wav" for name in names]
    infos = [get_audio_info(video_file) for video_file in video_files]
    sizes = [int(info['streams'][0].get('nb_frames', 0)) for info in infos]
    data.append([[name, str(video_file), str(audio_file), size] for name, video_file, audio_file, size in list(zip(names, video_files, audio_files, sizes))])
    return data
    

def get_label(csv_path):
    df = pd.read_csv(csv_path)
    name = df.loc[:, "file_name"]
    label = df.loc[:, "Thinking"].values.tolist()
    
    labels = [json.dumps(classes[int(i)]) for i in label]
    
    return labels

def get_video_info(video_path):
    """获取视频基本信息"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return info

def get_audio_info(file_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration",
        "-of", "json",
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    return info["streams"][0]
         
if __name__ == "__main__":
    anno_path = "/n/work1/muyun/Dataset/datasets_v3/annotation.csv"
    video_dir = "/n/work1/muyun/Dataset/datasets_v3/videos"
    audio_dir = "/n/work1/muyun/Dataset/datasets_v3/audios"
    labels = get_label(anno_path)
    data = get_data(anno_path, video_dir, audio_dir)[0]
    
    train_start_date = datetime.strptime("20231106", "%Y%m%d")
    train_end_date = datetime.strptime("20231220", "%Y%m%d")
    val_start_date = datetime.strptime("20231221", "%Y%m%d")
    val_end_date = datetime.strptime("20240118", "%Y%m%d")
    test_start_date = datetime.strptime("20240119", "%Y%m%d")
    test_end_date = datetime.strptime("20240124", "%Y%m%d")
    
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    print(len(data))
    print(len(labels))
    
    for d, l in zip(data, labels):
        name = d[0]
        date_str = name[:8]
        file_date = datetime.strptime(date_str, "%Y%m%d")
        if train_start_date <= file_date <= train_end_date:
            train_data.append(d)
            train_labels.append(l)
        elif val_start_date <= file_date <= val_end_date:
            val_data.append(d)
            val_labels.append(l)
        else:
            test_data.append(d)
            test_labels.append(l)
            
    save_data(train_data, train_labels, "train")
    save_data(test_data, test_labels, "test")
    save_data(val_data, val_labels, "val")