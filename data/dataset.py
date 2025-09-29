import logging
import json
import copy
from typing import Any, Sequence, Dict
import numpy as np
import random
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, WhisperProcessor, PreTrainedTokenizer
import torchaudio

from .mm_utils import process_audio, process_audio_file, process_video, load_video

logger = logging.getLogger(__name__)


def preprocess_simple(
    sources: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    if roles[sources[0]["from"]] != "user":
        # Skip the first one if it is not from human
        sources = sources[1:]
    message = sources[0]['value']
    target = sources[1]['value']
    input_ids = tokenizer(message, return_tensors='pt').input_ids[0]
    labels = tokenizer(target, return_tensors='pt').input_ids[0]
    labels = torch.cat((labels, torch.tensor([tokenizer.eos_token_id]).long()))
    
    return dict(input_ids=input_ids, labels=labels)


class SilenceDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        data_args
       ):
        self.data_args = data_args
        self.subset = data_args.subset
        self.video_processor = data_args.video_processor
        self.audio_processor = data_args.audio_processor
        self.tokenizer = tokenizer
        
        self.list_data_dict = json.load(open(data_path, "r"))
        
        
        logger.info(f"Initialized SilenceDataset with {len(self)} samples.")


    def __getitem__(self, index) -> Dict[str, Any]:
        sources = self.list_data_dict[index]
        
        if "audio" in sources:
            audio_file = sources["audio"]
            try:
                if "BEATs" in self.data_args.audio_tower:
                    audio = process_audio_file(audio_file)
                else:
                    audio = process_audio(audio_file, self.audio_processor)
            except Exception as e:
                print(e)
                
            
        if "video" in sources:
            video_file = sources["video"]
            try:
                if "AV-HuBERT" in self.data_args.vision_tower:
                    video = self.video_processor(video_file, None)["video_source"]
                elif "marlin" in self.data_args.vision_tower:
                    frames, _ = load_video(video_file)
                    frames = torch.FloatTensor(frames.copy())
                    video = self.video_processor(frames)
                elif "clip" in self.data_args.vision_tower or "siglip" in self.data_args.vision_tower:
                    video = process_video(video_file, processor=self.video_processor)
            except Exception as e:
                print(e)
                
        class_ = 0 if "stopped" in sources["conversations"][1]["value"] else 1
        
        sources = copy.deepcopy(sources["conversations"])
                
        data_dict = preprocess_simple(sources, self.tokenizer)
        data_dict["class"] = class_

        if 'video' in self.list_data_dict[index] and 'audio' in self.list_data_dict[index]:
            data_dict['video'] = video
            data_dict['audio'] = audio
        elif 'video' in self.list_data_dict[index]:
            data_dict['video'] = video
        elif 'audio' in self.list_data_dict[index]:
            data_dict['audio'] = audio
        
        return data_dict

    def __len__(self):
        return len(self.list_data_dict)
    
    

if __name__ == "__main__":
    from transformers import CLIPImageProcessor, WhisperProcessor
    from .collector import DataCollatorForSilenceDataset
    video_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    audio_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    class config:
        data_path="/n/work1/muyun/Dataset/datasets/custom_sft/config_v.json"
        subset = 'train'
        sample_rate = 16000
        noise_fn = "/misc/home/muyun/VScode/project/LLM/MMS-LLaMA-original/MMS-LLaMA/noise/babble_noise.wav"
        noise_prob = 0
        video_processor = video_processor
        audio_processor = audio_processor 
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    data_args = config()
    data = SilenceDataset(data_args.data_path, tokenizer, data_args)
    print(data[0])
    # d = DataCollatorForSilenceDataset(tokenizer)
    # dataloader = DataLoader(data, batch_size=4, num_workers=1, collate_fn=d)
    
    # print(next(enumerate(dataloader)))

    
