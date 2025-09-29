import os
from pathlib import Path
import re
import torch
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, 
                          AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline)

def ASR(file_path):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        device_map = "auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device_map="auto",
        generate_kwargs={"language":"japanese"}
    )

    result = pipe(file_path, return_timestamps=True, generate_kwargs={"language":"japanese"})
    return result

def save_text(input_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{audio_name}.txt")
    print(output_path)
    chunks = ASR(input_path)["chunks"]
    
    with open(output_path, "w") as f:
        for chunk in chunks:
            text = chunk['text']
            print(text)
            f.write(text + '\n')
    print(f"Save to {output_path}")
        
def batch_convert(input_dir, output_dir, extensions=None):
    if extensions is None:
        extensions = ['.wav']
    
    # 遍历目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                input_path = os.path.join(root, file)
                save_text(input_path, output_dir)
        
if __name__ == "__main__":
    root_path = "/n/work1/muyun/SemiAutonomous/segment5"
    p = Path(root_path)
    video_dir = [x for x in p.iterdir() if x.is_dir()]
    audio_dir = [x.joinpath("_audio") for x in video_dir]
    
    for dir in audio_dir:
        input_dir = str(dir)
        output_dir = dir.parent.joinpath("_text")
        batch_convert(input_dir, output_dir)