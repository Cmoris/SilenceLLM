from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import pathlib
import os

import torch

from transformers import (
    AutoTokenizer,
    Trainer,
    BitsAndBytesConfig
)
import transformers

from model import *
from data import SilenceDataset, DataCollatorForSilenceDataset
from silence_trainer import (SilenceTrainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    safe_save_model_for_hf_trainer
)


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
    llm_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")
    modality: List[str] = field(default_factory=list)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    tune_vision_tower : bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_hidden_size : int = field(default=None)
    # Audio tower Arguments
    audio_tower: Optional[str] = field(default=None)
    tune_audio_tower: bool = field(default=False)
    mm_hidden_size_a : int = field(default=None)
    # Modality fusion methods
    modality_fuse : str = field(default="concat")
    # Speech rate predictor
    sr_predictor : Optional[str] = field(default=None)
    sr_predictor_layers : Optional[int] = field(default=2)
    # Qformer
    use_qformer : bool = field(default=True)
    qformer_model : str = field(default="bert-large-uncased")
    qformer_layers : int = field(default=2)
    qformer_dim : int = field(default=1024)
    queries_per_sec : int = field(default=3)

@dataclass
class AVHubertArguments:
    w2v_path: str = field(default=None)
    apply_mask: bool = field(default=False)
    mask_selection: str = field(default="static")
    mask_length: int = field(default=10)
    mask_other: int = field(default=0)
    mask_prob: float = field(default=0.75)
    mask_channel_selection: str = field(default="static")
    mask_channel_length: int = field(default=64)
    mask_channel_other: int = field(default=0)
    mask_channel_prob: float = field(default=0.5)
    av_layerdrop: float = field(default=0.1)
    av_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.1)
    attention_dropout: float = field(default=0.0)
    feature_grad_mult: float = field(default=1.0)
    freeze_finetune_updates: int = field(default=0)
    
    
@dataclass
class DataArguments:
    # Path Arguments
    training_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    subset : str = field(default=None, metadata={"help": "Should be 'train', 'test', or 'val'"})

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    optim: str = field(default="adamw_torch")
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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = SilenceDataset(data_path=data_args.training_data_path,
                                   data_args=data_args,
                                   tokenizer=tokenizer)
    
    eval_dataset = SilenceDataset(data_path=data_args.val_data_path,
                                   data_args=data_args,
                                   tokenizer=tokenizer)
    
    data_collator = DataCollatorForSilenceDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

    
def train():
    global local_rank
    set_seed(42)
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    bnb_model_from_pretrained_args = {}
    bnb_model_from_pretrained_args.update(
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ).to_dict()
    )
    
    config = SilenceLLMConfig(
        vision_tower = model_args.vision_tower,
        mm_vision_select_layer = model_args.mm_vision_select_layer,
        mm_vision_select_feature = model_args.mm_vision_select_feature,
        audio_tower = model_args.audio_tower,
        use_qformer = model_args.use_qformer,
        qformer_model = model_args.qformer_model,
        qformer_layers = model_args.qformer_layers,
        qformer_dim = model_args.qformer_dim,
        queries_per_sec = model_args.queries_per_sec,
        modality_fuse = model_args.modality_fuse,
        sr_predictor = model_args.sr_predictor,
        sr_predictor_layers = model_args.sr_predictor_layers,
        llm_path = model_args.llm_path,
        lora_rank = training_args.lora_r,
        lora_alpha = training_args.lora_alpha,
        target_modules = training_args.target_modules,
        bnb_config = bnb_model_from_pretrained_args
    )
    
    model = SilenceLLMModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    
    vision_tower = model.get_vision_tower()
    audio_tower = model.get_audio_tower()
    
    data_args.video_processor = vision_tower.image_processor
    data_args.audio_processor = audio_tower.audio_processor
    
    print("Current model:", model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    model.tie_weights()
    trainer = Trainer(model=model, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, pathlib.Path(training_args.output_dir)/'non_lora_trainables.bin')
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()
    
    