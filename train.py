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
from silence_trainer import (SilenceTrainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    safe_save_model_for_hf_trainer
)
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
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    tune_adapter_llm: bool = field(default=False)
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
    
    eval_dataset = SilenceDataset(data_path=data_args.val_data_path,
                                   data_args=data_args,
                                   tokenizer=tokenizer)
    
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

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path)

    config._attn_implementation = attn_implementation

    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=compute_dtype,
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError

    model.config.use_cache = False
    model.config.subset = model_args.subset = data_args.subset
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.model_base = model_args.model_path

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split('.'),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
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
        
        if "BEATs" not in model_args.audio_tower:
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
        
    
    # if training_args.bits in [4,8]:
    #     from peft.tuners.lora import LoraLayer
    #     for name, module in model.named_modules():
    #         if isinstance(module, LoraLayer):
    #             if training_args.bf16:
    #                 module = module.to(torch.bfloat16)
    #         if 'norm' in name:
    #             module = module.to(torch.float32)
    #         if 'lm_head' in name or 'embed_tokens' in name:
    #             if hasattr(module, 'weight'):
    #                 if training_args.bf16 and module.weight.dtype == torch.float32:
    #                     module = module.to(torch.bfloat16)

    print("Current model:", model)
    data_args.vision_tower = model_args.vision_tower
    data_args.audio_tower = model_args.audio_tower
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    model.print_trainable_parameters()
    
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

    trainer = SilenceTrainer(model=model, 
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

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
