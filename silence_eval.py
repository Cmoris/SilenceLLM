import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments
import numpy as np
import evaluate
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from model import SilenceLLMModel, SilenceLLMConfig
from data import SilenceDataset, DataCollatorForSilenceDataset

@dataclass
class DataArguments:
    # Path Arguments
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    subset: str = field(default=None)

@dataclass
class ModelLoadArguments:
    llm_path: str = field(default=None)
    model_path: str = field(default=None, metadata={"help": "Path to the pretrained model or checkpoint"})
    

def test():
    parser = HfArgumentParser((ModelLoadArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 设置日志，方便查看信息 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"加载模型于: {model_args.model_path}")
    logging.info(f"加载测试数据于: {data_args.test_data_path}")

    # --- 加载模型、Tokenizer 和多模态处理器 ---
    config = SilenceLLMConfig.from_pretrained(model_args.model_path)
    model = SilenceLLMModel(config)
    
    if Path.exists(Path(model_args.model_path)/'non_lora_trainables.bin'):
        non_lora_trainables = torch.load(Path(model_args.model_path)/'non_lora_trainables.bin', map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(model_args.model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)
    
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_args.model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 vision 和 audio 处理器
    vision_tower = model.get_vision_tower()
    audio_tower = model.get_audio_tower()
    data_args.video_processor = vision_tower.image_processor
    data_args.audio_processor = audio_tower.audio_processor

    # --- 准备测试数据集 ---
    logging.info("准备测试数据集...")
    test_dataset = SilenceDataset(
        data_path=data_args.test_data_path,
        data_args=data_args,
        tokenizer=tokenizer
    )
    data_collator = DataCollatorForSilenceDataset(tokenizer=tokenizer)

    # --- 定义评估指标计算函数 ---
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    # --- 配置 Trainer ---
    # 对于纯评估，我们只需要传入少数几个参数
    # ！！！predict_with_generate 依然至关重要！！！
    training_args.predict_with_generate = True
    training_args.generation_max_length = 512 # 与训练时保持一致或按需修改
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,  # 注意这里传入的是 eval_dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 执行评估 ---
    logging.info("开始评估...")
    results = trainer.evaluate(eval_dataset=test_dataset) # 也可以直接调用 evaluate()

    # --- 打印结果 ---
    logging.info("评估完成！")
    print("="*30)
    print("      评估结果      ")
    print("="*30)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*30)


if __name__ == "__main__":
    test()