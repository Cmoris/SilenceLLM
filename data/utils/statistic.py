import json

# with open('/misc/home/muyun/VScode/project/LLM/Silence/data/utils/labels.jsonl', 'r') as f:
#     lines = f.readlines()
#     speech_segments = []
#     for line in lines:
#         data = json.loads(line)
#         silence_segments = data['silence_segments']
#         end = silence_segments[0]["end"]
#         for i, seg in enumerate(silence_segments[1:]):
#             start = seg["start"]
#             speech_segment =  start - end
#             end = seg["end"]
#             speech_segments.append(speech_segment)
# import ipdb; ipdb.set_trace()
# s = 0
# for seg in speech_segments:
#     s += seg
# avg = s/len(speech_segments)
# print(avg)

# def load_data(path):
#     with open(path, 'r') as f:
#         json_data = json.load(f)
#     stopped, thinking = 0, 0
#     for data in json_data:
#         label = data["conversations"][1]["value"]
#         if "stopped" in label:
#             stopped += 1
#         else:
#             thinking += 1
#     print("stopped: ", stopped)
#     print("thinking: ", thinking)
        
# if __name__ == "__main__":
#     train_path = "/n/work1/muyun/Dataset/datasets_v3/labels/config_mm_train.json"
#     val_path = "/n/work1/muyun/Dataset/datasets_v3/labels/config_mm_val.json"
#     test_path = "/n/work1/muyun/Dataset/datasets_v3/labels/config_mm_test.json"
    
#     load_data(train_path)
#     load_data(val_path)
#     load_data(test_path)
sr = 0
with open("/home/muyun/VScode/project/ASR/data/transcripts/20250825_ja.jsonl", 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        text = data["text"]
        sr += len(text) / 5
        
