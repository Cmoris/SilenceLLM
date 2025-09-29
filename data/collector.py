from typing import Dict, List, Any
import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollatorForSilenceDataset:
    def __init__(self, tokenizer, subset="train", padding_value=0, return_tensors="pt", device="cuda"):
        self.tokenizer = tokenizer
        self.padding_value = padding_value
        self.return_tensors = return_tensors
        self.device = device
        self.subset = subset

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ==================== 文本部分 ====================
        input_ids = [feature["input_ids"].detach().clone().squeeze(0) for feature in features]
        labels = [feature["labels"].detach().clone().squeeze(0) for feature in features]
        classes = torch.tensor([feature["class"] for feature in features]).to(torch.long)

        # 使用 pad_sequence 填充文本
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "classes": classes
        }

        if "audio" in features[0]:
        # ==================== 音频部分 ====================
            audio_batch = [feature["audio"].squeeze(0) for feature in features]  # 去掉 batch 维度
            # 保存原始长度用于生成 mask
            audio_lengths = [audio.shape[0] for audio in audio_batch]
            # padding
            audio_padded = pad_sequence(audio_batch, batch_first=True, padding_value=0.0)
            batch["audio"] = audio_padded

            # 生成 audio padding mask (1 for valid, 0 for padding)
            max_audio_len = audio_padded.shape[1]
            audio_masks = []
            for length in audio_lengths:
                mask = torch.ones(length, dtype=torch.long)
                padding_mask = torch.cat([mask, torch.zeros(max_audio_len - length, dtype=torch.long)])
                audio_masks.append(padding_mask)
            batch["audio_mask"] = torch.stack(audio_masks)  # (B, T_audio)

        if "video" in features[0]:
            # ==================== 视频部分 ====================
            video_batch = [feature["video"].squeeze(0) for feature in features]
            # 保存原始长度用于生成 mask
            video_lengths = [video.shape[0] for video in video_batch]
            # padding
            video_padded = pad_sequence(video_batch, batch_first=True, padding_value=0.0)
            batch["video"] = video_padded

            # 生成 video padding mask (1 for valid, 0 for padding)
            max_video_len = video_padded.shape[1]
            video_masks = []
            for length in video_lengths:
                mask = torch.ones(length, dtype=torch.long)
                padding_mask = torch.cat([mask, torch.zeros(max_video_len - length, dtype=torch.long)])
                video_masks.append(padding_mask)
            batch["video_mask"] = torch.stack(video_masks)  # (B, T_video)
        
        if self.subset != "train":
            for k, v in batch.items():
                batch[k] = v.to(device=self.device)
            
        source = {'source':batch}
        
        return source
    
