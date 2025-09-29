from transformers import HubertModel, HubertConfig, AutoProcessor
import torch
import torch.nn as nn

import einops

class Hubert(nn.Module):
    def __init__(self, audio_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
             
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = HubertConfig.from_pretrained(audio_tower)
            
    def load_model(self):
        self.audio_tower = HubertModel.from_pretrained(self.audio_tower_name)
        self.hubert_config = HubertConfig.from_pretrained(self.audio_tower_name)
        self.audio_processor = AutoProcessor.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)
        self.is_loaded = True
        
    
    
    @torch.no_grad
    def forward(self, audio):
        logits = self.audio_tower(audio)
        return logits
    
    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.hubert_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    
if __name__ == "__main__":
    from scipy.io import wavfile
    model = Hubert("facebook/hubert-large-ls960-ft", delay_load=False)
    print(model)
    print(model.config)
    audio_path = "/n/work1/muyun/Dataset/datasets_v3/audios/20231106_01_1.wav"
    sr, wav = wavfile.read(audio_path)
    audio_inputs = model.audio_processor(wav, sampling_rate=sr, return_tensors="pt")
    audio_feats = audio_inputs.input_features if hasattr(audio_inputs, "input_features") else audio_inputs.input_values
    print(audio_feats.size())
    