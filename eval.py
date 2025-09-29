import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    PretrainedConfig, 
    BitsAndBytesConfig,
    StoppingCriteria)
import transformers

from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json

import sys
sys.path.append('./')
from model import *
from model.submodels.projector import load_mm_projector
from model.submodels.modules import build_audio_tower
from data import process_video, process_audio, SilenceDataset, DataCollatorForSilenceDataset


@dataclass
class DataArguments:
    # Path Arguments
    training_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    subset : str = field(default=None, metadata={"help": "Should be 'train', 'test', or 'val'"})
    output_dir : str = field(default=None)
    
@dataclass
class ModelArguments:
    num_beams : int = field(default=None)
    temperature : float = field(default=None)
    model_path : str = field(default=None) 
    model_base : str = field(default=None) 
    model_type : str = field(default="SilenceQwen3")
    

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, start_len):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = start_len
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        import ipdb; ipdb.set_trace()
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def model_init(model_path=None, **kwargs):
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    processor = {
        'video': partial(process_video, processor=processor),
        'audio': process_audio,
    }

    return model, processor, tokenizer

def load_pretrained_model(model_path, model_base, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    config = VLLMConfigs[model_args.model_type].from_pretrained(model_path)
    config.subset = 'test'
    # judge model type
    model_type = config.model_type

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token = "<|end_of_text|>"
    print('Loading SilenceQwen3 from base model...')
    model = VLLMs[model_args.model_type].from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
    
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    
    if Path.exists(Path(model_path)/'non_lora_trainables.bin'):
        non_lora_trainables = torch.load(Path(model_path)/'non_lora_trainables.bin', map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    
    audio_processor = None
    video_processor = None
    if getattr(model.config, "mm_audio_tower", None) is not None:
        if "BEATs" in getattr(model.config, "mm_audio_tower", None):
            audio_tower = model.get_audio_tower()
            beats, beats_cfg = build_audio_tower(model.config)
            audio_tower.load_state_dict(beats.state_dict(), strict=False)
            audio_processor = None
        else:
            audio_tower = model.get_audio_tower()
            audio_tower.load_model()
            audio_processor = audio_tower.audio_processor
    if getattr(model.config, "mm_vision_tower", None) is not None:
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

    if getattr(model.config, "use_sr_predictor", False):
        sr_predictor = model.get_sr_predictor()
        sr_predictor.load_model()
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 1024
        
    return tokenizer, model, config, video_processor, audio_processor, context_len


def load_data(data_args, tokenizer, video_processor, audio_processor):
    data_args.video_processor = video_processor
    data_args.audio_processor = audio_processor
    
    dataset = SilenceDataset(data_args.test_data_path, tokenizer=tokenizer, data_args=data_args)
    
    collator = DataCollatorForSilenceDataset(tokenizer=tokenizer, subset=data_args.subset)
    
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collator)
    
    return dataloader


def mm_infer(dataloader, model, tokenizer, output_dir, **kwargs):
    model.generation_config.pad_token_id = tokenizer("<|finetune_right_pad_id|>").input_ids[1]
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval().cuda()
    num_sentences = 0
    result_dict = {"model_path":kwargs['model_path'], "ref": [], "hypo": [], "instruction": []}
    
    do_sample = kwargs.get('do_sample', True)
    num_beams = kwargs.get('num_beams', 5)
    temperature = kwargs.get('temperature', 0.7 if do_sample else 0.0)
    max_new_tokens = kwargs.get('max_new_tokens', 100)
    import ipdb; ipdb.set_trace()
    for i, source in enumerate(dataloader):
        source = source["source"]
        with torch.inference_mode():
            best_hypo = model.generate(
                source=source,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        best_hypo = tokenizer.batch_decode(
            best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for j in range(len(source['labels'])):            
            target = source["labels"][j].masked_fill(
                source["labels"][j] == -100, 0
            )
            
            ref_sent = tokenizer.decode(target.int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['ref'].append(ref_sent.strip())
            hypo_str = best_hypo[j].strip()
            
            instruction = tokenizer.decode(source["input_ids"][j].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['instruction'].append(instruction)
            result_dict['hypo'].append(hypo_str)
            
            print(f"ref: {ref_sent.strip()}")
            print(f"hypo: {hypo_str}\n")
            
            num_sentences += 1
    Path(output_dir).mkdir(exist_ok=True)
    output_json = Path(output_dir) / "result.json"
    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(result_dict, f_out, indent=2 ,ensure_ascii=False)
            
    return result_dict

    

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    tokenizer, model, config, video_processor, audio_processor, context_len = load_pretrained_model(model_path=model_args.model_path, 
                                                                                            model_base=model_args.model_base)
    data_args.vision_tower = config.mm_vision_tower if hasattr(config, "mm_vision_tower") else None
    data_args.audio_tower = config.mm_audio_tower if hasattr(config, "mm_audio_tower") else None
    dataloader = load_data(data_args=data_args, tokenizer=tokenizer, video_processor=video_processor, audio_processor=audio_processor)
    
    result = mm_infer(dataloader, 
                      model=model, 
                      tokenizer=tokenizer, 
                      output_dir=data_args.output_dir,
                      num_beams = model_args.num_beams,
                      temperature = model_args.temperature,
                      model_path = model_args.model_path)
    

    