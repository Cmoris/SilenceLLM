#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${1:-1}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
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

export CUDA_VISIBLE_DEVICES=4

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    train_v2.py \
    --training_data_path /n/work1/muyun/Dataset/datasets_v3/labels_no_val/config_mm_train.json  \
    --subset train \
    --model_type SilencePerceiver \
    --dataloader_num_workers 4 \
    --vocab_size 151936 \
    --hidden_size 1024 \
    --depth 6 \
    --max_seq_len 2048 \
    --num_latents 256 \
    --latent_dim 512 \
    --cross_heads 1 \
    --latent_heads 8 \
    --cross_dim_head 64 \
    --latent_dim_head 64 \
    --weight_tie_layers False \
    --num_classes 2 \
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
    --vision_tower google/siglip2-base-patch16-224 \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --audio_tower openai/whisper-medium \
    --pretrained_audio_tower None \
    --mm_projector_v_type stp_connector \
    --mm_projector_a_type conv \
    --tune_mm_mlp_adapter True \
    --output_dir /n/work1/muyun/Model/MMS_LLAMA/audio_video/SilencePerceiver-siglip2-no_sr-concat-float32 \
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
