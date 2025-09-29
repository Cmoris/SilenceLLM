import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
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
    model = VLLMs[model_args.model_type].from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    
    print('Model is loaded...')
    
    audio_processor = None
    video_processor = None
    if getattr(model.config, "mm_audio_tower", None) is not None:
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
    model.eval().cuda()
    num_sentences = 0
    result_dict = {"model_path":kwargs['model_path'], "ref": [], "hypo": [], "instruction": []}
    
    for i, source in enumerate(dataloader):
        source = source["source"]
        with torch.inference_mode():
            hidden_states = model(source).hidden_states
            prob = F.softmax(hidden_states, dim=1)
            best_hypo = torch.max(prob, dim=1)[1]

        for j in range(len(source['classes'])):            
            target = source["classes"][j]
            
            result_dict['ref'].append(str(target))
            
            result_dict['hypo'].append(str(best_hypo[j]))
            
            print(f"ref: {target}")
            print(f"hypo: {best_hypo[j]}\n")
            
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
    dataloader = load_data(data_args=data_args, tokenizer=tokenizer, video_processor=video_processor, audio_processor=audio_processor)
    
    result = mm_infer(dataloader, 
                      model=model, 
                      tokenizer=tokenizer, 
                      output_dir=data_args.output_dir,
                      num_beams = model_args.num_beams,
                      temperature = model_args.temperature,
                      model_path = model_args.model_path)
    

    