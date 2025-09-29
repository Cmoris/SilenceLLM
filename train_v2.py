import re
import os
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from functools import partial

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
import evaluate
wer_metric = evaluate.load("wer")

import sys
sys.path.append('./')
from model import *
from data import SilenceDataset, DataCollatorForSilenceDataset

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="SilenceQwen3")
    model_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")
    vocab_size: int = field(default=151936)
    hidden_size: int = field(default=1024)
    num_classes: int = field(default=2)
    # PerceiverIO
    depth: int = field(default=6)
    max_seq_len: int = field(default=2048) 
    num_latents: int = field(default=256) 
    latent_dim: int = field(default=512) 
    cross_heads: int = field(default=1) 
    latent_heads: int = field(default=8) 
    cross_dim_head: int = field(default=64) 
    latent_dim_head: int = field(default=64) 
    weight_tie_layers: bool = field(default=False) 
    # Connector Arguments
    mm_projector_v_type: Optional[str] = field(default='linear')
    mm_projector_a_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_adapter_a: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    pretrained_vision_tower: Optional[str] = field(default=None)
    tune_vision_tower : bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_hidden_size_v : int = field(default=0)
    # Audio tower Arguments
    audio_tower: Optional[str] = field(default=None)
    pretrained_audio_tower: Optional[str] = field(default=None)
    tune_audio_tower: bool = field(default=False)
    mm_hidden_size_a : int = field(default=0)
    # Modality fusion methods
    modality_fuse : str = field(default="concat")
    # Speech rate predictor
    use_sr_predictor : Optional[bool] = field(default=True)
    sr_predictor : Optional[str] = field(default=None)
    sr_predictor_layers : Optional[int] = field(default=None)
    # Qformer
    use_qformer : bool = field(default=True)
    window_level_Qformer : bool = field(default=False)
    qformer_model : str = field(default="bert-large-uncased")
    qformer_layers : int = field(default=2)
    qformer_dim : int = field(default=1024)
    queries_per_sec : int = field(default=3)
    second_per_window : float = field(default=0.333333)
    second_stride : float = field(default=0.333333)
    mm_hidden_size : int = field(default=0)
    
    
@dataclass
class DataArguments:
    # Path Arguments
    training_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    subset : str = field(default=None, metadata={"help": "Should be 'train', 'test', or 'val'"})
    #Marlin processor
    image_crop_size: int = field(default=88)
    image_mean: float = field(default=0.0)
    image_std: float = field(default=255.0)
    #Processor
    video_processor = None
    audio_processor = None

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    target_modules : str = "q_proj.k_proj.v_proj.o_proj"
    
    save_lora_and_adapter : bool = field(default=True)
    
    weighted_sampler : bool = field(default=True)
    


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = SilenceDataset(data_path=data_args.training_data_path,
                                   data_args=data_args,
                                   tokenizer=tokenizer)
    
    # eval_dataset = SilenceDataset(data_path=data_args.val_data_path,
    #                                data_args=data_args,
    #                                tokenizer=tokenizer)
    
    data_collator = DataCollatorForSilenceDataset(tokenizer=tokenizer, subset=data_args.subset)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    # from transformers import BitsAndBytesConfig
    # bnb_model_from_pretrained_args.update(dict(
    #     # device_map={"": training_args.device},
    #     # BUG: High version transformers report error: 
    #     # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
    #     # load_in_4bit=training_args.bits == 4,
    #     # load_in_8bit=training_args.bits == 8,
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=training_args.double_quant,
    #         bnb_4bit_quant_type=training_args.quant_type,
    #         bnb_4bit_compute_dtype=compute_dtype
    #     )
    # ))

    config = VLLMConfigs[model_args.model_type](
        vocab_size=model_args.vocab_size,
        hidden_size=model_args.hidden_size,
        num_classes=model_args.num_classes,
        depth = model_args.depth,                   
        max_seq_len = model_args.max_seq_len,          # maximum sequence length
        num_latents = model_args.num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = model_args.latent_dim,            # latent dimension
        cross_heads = model_args.cross_heads,             # number of heads for cross attention. paper said 1
        latent_heads = model_args.latent_heads,            # number of heads for latent self attention, 8
        cross_dim_head = model_args.cross_dim_head,         # number of dimensions per cross attention head
        latent_dim_head = model_args.latent_dim_head,        # number of dimensions per latent self attention head
        weight_tie_layers = model_args.weight_tie_layers,
    )

    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = VLLMs[model_args.model_type](config=config)
    else:
        raise ValueError

    model.config.use_cache = False
    model.config.subset = model_args.subset = data_args.subset
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.model_base = model_args.model_path
    
    if training_args.bits == 32:
        model.to(dtype=torch.float32)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token = "<|end_of_text|>"
        
    pad_token_id = tokenizer("<|finetune_right_pad_id|>").input_ids[1]
    model.config.tokenizer_pad_token_id = pad_token_id
    
    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model_args.subset = data_args.subset
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float32, device=training_args.device)
        
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True
        
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector_v.parameters():
                p.requires_grad = True

        model.get_model().mm_projector_v.to(dtype=torch.float32, device=training_args.device)
    
        
    if model_args.audio_tower is not None:
        # initialize audio encoder + multi-modal projector
        model.get_model().initialize_audio_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        audio_tower = model.get_audio_tower()
        audio_tower.to(dtype=torch.float32, device=training_args.device)
        
        data_args.is_multimodal = True
        
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector_a.parameters():
                p.requires_grad = True

        model.get_model().mm_projector_a.to(dtype=torch.float32, device=training_args.device)
        
        data_args.audio_processor = audio_tower.audio_processor
    
    if model_args.sr_predictor is not None and getattr(model_args, "use_sr_predictor", True):
        model.get_model().initialize_sr_predictor(
            model_args=model_args,
            fsdp = training_args.fsdp
        )    
        
        sr_predictor = model.get_sr_predictor()
        sr_predictor.to(dtype=torch.float32, device=training_args.device)
                
        for param in sr_predictor.parameters():
            param.requires_grad = False

    if model_args.qformer_model is not None:
        if model_args.vision_tower is None:
            model.config.mm_hidden_size_v = 0
        elif model_args.audio_tower is None:
            model.config.mm_hidden_size_a = 0
        model.get_model().initialize_qformer(
            model_args=model_args,
            fsdp = training_args.fsdp
        )
        
        qformer = model.get_qformer()
        qformer.to(dtype=torch.float32, device=training_args.device)
       
        for param in qformer.parameters():
            param.requires_grad = True
        
        model.get_model().query_tokens.to(dtype=torch.float32, device=training_args.device)
        
        model.get_model().initialize_projector(
            model_args=model_args,
            fsdp = training_args.fsdp
        )
            
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
                
        model.get_model().mm_projector.to(dtype=torch.float32, device=training_args.device)
        

    print("Current model:", model)
    data_args.vision_tower = model_args.vision_tower
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    
    def compute_metrics(eval_pred, tokenizer):

        predictions, labels = eval_pred


        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels[labels == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_str_clean = [s.strip() for s in pred_str]
        label_str_clean = [s.strip() for s in label_str]

        wer_score = wer_metric.compute(predictions=pred_str_clean, references=label_str_clean)

        return {"wer": wer_score}
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    trainer = transformers.Trainer(model=model, 
                    tokenizer=tokenizer, 
                    args=training_args, 
                    **data_module, 
                    compute_metrics=compute_metrics_with_tokenizer)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

if __name__ == "__main__":
    train()
