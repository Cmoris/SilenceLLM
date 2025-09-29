#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${1:-1}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

export CUDA_VISIBLE_DEVICES=2

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    train.py \
    --training_data_path /n/work1/muyun/Dataset/datasets_v3/labels_no_val/config_a_train.json  \
    --subset train \
    --dataloader_num_workers 4 \
    --model_path Qwen/Qwen3-0.6B \
    --use_qformer True \
    --window_level_Qformer False \
    --qformer_model bert-large-uncased \
    --qformer_dim 1024 \
    --qformer_layers 2 \
    --queries_per_sec 3 \
    --second_per_window 0.333333 \
    --second_stride 0.333333 \
    --use_sr_predictor False \
    --sr_predictor /n/work1/muyun/Model/MMS_LLAMA/sr_predictor/checkpoint.pt \
    --sr_predictor_layers 2 \
    --modality_fuse concat \
    --audio_tower openai/whisper-medium \
    --pretrained_audio_tower None \
    --mm_projector_a_type conv \
    --tune_mm_mlp_adapter True \
    --model_max_length  1024\
    --bits 32 \
    --bf16 False \
    --fp16 False \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj.k_proj.v_proj.o_proj \
    --output_dir /n/work1/muyun/Model/MMS_LLAMA/audio_only/SilenceQwen3-0.6B-lora16-whisper-no_sr-audio_only-float32_2 \
    --adam_beta2 0.98 \
    --label_smoothing_factor 0.1 \
    --learning_rate 1e-5 \
    --weight_decay  0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --weighted_sampler False \
    --logging_steps 1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    # --eval_strategy epoch \
    # --load_best_model_at_end True \
    # --metric_for_best_model wer
