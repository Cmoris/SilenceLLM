from avhubert import AV2TextForConditionalGeneration, AV2TextConfig, load_feature 

import torch
import torch.nn as nn

import einops

class AVHubert(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
             
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AV2TextConfig.from_pretrained(self.vision_tower_name)
            
    def load_model(self):
        # models, config, _ = checkpoint_utils.load_model_ensemble_and_task([self.ckpt_path], arg_overrides=self.arg_overrides, strict=False)
        
        # avhubert = HubertEncoderWrapper(models[0])
        # avhubert.w2v_model.remove_pretraining_modules()

        avhubert_config = AV2TextConfig.from_pretrained(self.vision_tower_name)
        avhubert = AV2TextForConditionalGeneration.from_pretrained(self.vision_tower_name).get_encoder()
        
        self.vision_tower = avhubert
        self.avhubert_config = avhubert_config
        self.vision_tower.requires_grad_(False)

        self.video_processor = load_feature
            
        self.is_loaded = True
    
    @torch.no_grad
    def forward(self, source):

        video = einops.rearrange(source["video"], 'b t c h w -> b c t h w')
        avhubert_source = {'audio': None, 'video': video}
        video_features, padding_mask = self.vision_tower.extract_finetune(source=avhubert_source, padding_mask=source['video_mask'])
        import ipdb; ipdb.set_trace()
        return video_features, padding_mask
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.config.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.avhubder_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    
if __name__ == "__main__":
    
    # source = {"video":torch.zeros((4,209,1,88,88)), "video_mask":torch.ones((4,209))}
    # model = AVHubert(vision_tower=f"nguyenvulebinh/AV-HuBERT-MuAViC-en")
    

    # print(model(source))
    print(load_feature(video_path="/n/work1/muyun/Dataset/datasets_v3/videos_1/20231106_01_1.mp4", audio_path=None)["video_source"].size())