export CUDA_VISIBLE_DEVICES=3

python eval.py \
    --model_path  "/n/work1/muyun/Model/silence/SilenceLlama3.2-3B-lora16-siglip2-beats-no_sr-no_q-concat-float32"\
    --model_type SilenceLlama3 \
    --model_base  /n/work1/muyun/Model/Llama3.2-3B-Instruct-hf \
    --num_beams 5 \
    --temperature 0.3 \
    --test_data_path  /n/work1/muyun/Dataset/datasets_v3/labels_no_val/config_mm_test.json \
    --subset test \
    --output_dir /home/muyun/VScode/project/LLM/Silence/result/llama/result-siglip2-beats-no_sr-no_q-concat

