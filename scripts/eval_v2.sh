export CUDA_VISIBLE_DEVICES=0

python eval_v2.py \
    --model_path  "/n/work1/muyun/Model/MMS_LLAMA/audio_video/SilencePerceiver-siglip2-no_sr-concat-float32/checkpoint-2930"\
    --model_type SilencePerceiver \
    --model_base Qwen/Qwen3-0.6B \
    --test_data_path  /n/work1/muyun/Dataset/datasets_v3/labels_no_val/config_mm_test.json \
    --subset test \
    --output_dir /misc/home/muyun/VScode/project/LLM/Silence/result/perceiver/result-siglip2-no_sr-concat
