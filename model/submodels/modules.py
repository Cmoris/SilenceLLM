import torch
import torch.nn as nn
from typing import Any, Optional
from dataclasses import dataclass, asdict
from argparse import Namespace

from torchvision import transforms
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
    Siglip2VisionModel, Siglip2ImageProcessor, Siglip2VisionConfig,
    WhisperForConditionalGeneration, WhisperConfig, WhisperFeatureExtractor,
    HubertModel, HubertConfig,
    AutoConfig, AutoModel, AutoProcessor,AutoFeatureExtractor
)

import einops

from types import SimpleNamespace

from .beats.BEATs import BEATsConfig, BEATs
from .beats.Tokenizers import Tokenizers, TokenizersConfig
from .Qformer import BertConfig, BertLMHeadModel
from .Transformer import TransformerEncoder
from .avhubert import AV2TextForConditionalGeneration, AV2TextConfig, load_feature 
     
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        # create a list of layers
        self.layers = nn.ModuleList([
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        ])
    
    def forward(self, x):
        # iterate through all the layers
        for layer in self.layers:
            x = layer(x)
        return x         
          
class Multimodal_Attention(nn.Module):
    def __init__(self, qdim, kdim, num_heads):
        super(Multimodal_Attention, self).__init__()
        # create a list of layers
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=qdim, kdim=kdim, vdim=kdim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(qdim)
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=qdim, kdim=kdim, vdim=kdim, num_heads=num_heads)
    
    def forward(self, audio_feature, visual_feature):
        # iterate through all the layers
        
        x, _ = self.mha0(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x = x + visual_feature
        x = self.layer_norm(x)
        x2, _ = self.mha1(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x2 = x + x2

        return x2
    
class MMClassifierHead(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        self.proj = nn.Linear(d_model, length)

    def forward(self, hidden):
        # hidden: [B, T, d_model]
        logits = self.proj(hidden)  # [B, T, N]
        return logits
  
        

@dataclass
class SpeechRatePredictorConfig:
    """Configuration for the Speech_Rate_Predictor model."""
    num_layers: int = 2
    input_dim: int = 1024
    encoder_embed_dim: int = 256
    encoder_ffn_embed_dim: int = 1024
    encoder_attention_heads: int = 4
    conv_pos: int = 128
    conv_pos_groups: int = 16
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.1
    encoder_layerdrop: float = 0.1
    activation_fn: str = 'gelu'
    layer_norm_first: bool = True
    
class Speech_Rate_Predictor(nn.Module):
    """
    Speech Rate Predictor model that encapsulates its own weight loading logic.

    Args:
        config (SpeechRatePredictorConfig): The configuration for the model architecture.
        checkpoint_path (Optional[str]): Path to the pretrained weights checkpoint. Required if not delay_load.
        delay_load (bool): If True, the model skeleton is created but weights are not loaded until
                           the first forward pass or a manual call to `load_model()`.
        freeze_on_load (bool): If True, freezes the model parameters after loading the weights.
    """
    def __init__(self,
                 config,
                 checkpoint_path: Optional[str] = None,
                 delay_load: bool = False,
                 freeze_on_load: bool = True):
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.freeze_on_load = freeze_on_load
        self.is_loaded = False
        
        transformer_args = SimpleNamespace(**asdict(config))
        transformer_args.encoder_layers = transformer_args.num_layers

        self.sr_token = nn.Parameter(torch.zeros(1, 1, config.encoder_embed_dim))
        self.input_proj = nn.Linear(config.input_dim, config.encoder_embed_dim)
        self.encoder = TransformerEncoder(transformer_args)
        self.sr_predictor_head = nn.Linear(config.encoder_embed_dim, 1)
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.sr_token)

        # 2. 根据 delay_load 决定是否立即加载权重
        if not delay_load:
            if self.checkpoint_path is None:
                raise ValueError("Must provide 'checkpoint_path' when 'delay_load' is False.")
            self.load_model()

    def load_model(self):
        """
        Loads the pretrained weights from the specified checkpoint path.
        This method is idempotent; it will not reload weights if already loaded.
        """
        if self.is_loaded:
            return
        
        if not self.checkpoint_path:
            raise ValueError("Cannot load model: 'checkpoint_path' was not provided during initialization.")

        print(f"Loading Speech_Rate_Predictor weights from {self.checkpoint_path}")
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")['model']

        # 清理权重字典的键名 (例如，去掉 "sr_predictor." 前缀)
        prefix_to_strip = "sr_predictor."
        prefix_len = len(prefix_to_strip)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_strip):
                original_key = k[prefix_len:]
                # 键名映射 (与之前的版本一样，根据需要调整)
                if original_key == 'linear.weight': new_key = 'input_proj.weight'
                elif original_key == 'linear.bias': new_key = 'input_proj.bias'
                elif original_key == 'sr_predictor.weight': new_key = 'sr_predictor_head.weight'
                elif original_key == 'sr_predictor.bias': new_key = 'sr_predictor_head.bias'
                else: new_key = original_key
                cleaned_state_dict[new_key] = v.to(torch.float32)

        self.load_state_dict(cleaned_state_dict, strict=True)
        print("Weights loaded successfully.")

        # 根据初始化时的设置决定是否冻结参数
        if self.freeze_on_load:
            for param in self.parameters():
                param.requires_grad = False
            self.eval() # 切换到评估模式

        self.is_loaded = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass. Automatically loads weights on the first call if not already loaded.
        """
        if not self.is_loaded:
            self.load_model()
        
        x = self.input_proj(features)
        batch_size = x.size(0)
        sr_token_expanded = self.sr_token.expand(batch_size, -1, -1)
        x = torch.cat([sr_token_expanded, x], dim=1)
        x, _ = self.encoder(x)
        sr_token_output = x[:, 0, :]
        sr_prediction = self.activation(self.sr_predictor_head(sr_token_output))
        
        return sr_prediction

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the model's parameters."""
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the model's parameters."""
        return next(self.parameters()).device


### Video Encoder ###
class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            video_length = len(images)
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            image_features = torch.cat(image_features, dim=0)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', t=video_length)
        else:
            batch_size = images.size(0)
            frames = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            image_forward_outs = self.vision_tower(frames.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', b=batch_size)
            
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            video_length = len(images)
            image_features = []
            for image in images:
                # image = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            image_features = torch.cat(image_features, dim=0)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', t=video_length)
        else:
            batch_size = images.size(0)
            frames = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            image_forward_outs = self.vision_tower(frames.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', b=batch_size)
            
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
    
    
class Siglip2VisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Siglip2VisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = Siglip2ImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = Siglip2VisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            video_length = len(images)
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            image_features = torch.cat(image_features, dim=0)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', t=video_length)
        else:
            batch_size = images.size(0)
            frames = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            image_forward_outs = self.vision_tower(frames.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = einops.rearrange(image_features, '(b t) h w -> b t h w', b=batch_size)
            
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
    
    
class MARLINVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.image_crop_size = 224
        self.image_mean = 0.421
        self.image_std = 0.165
        self.subset = args.subset
        self.chunk_size = 16
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                            vision_tower,
                            trust_remote_code=True
                        )
            
    def load_model(self):
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self.vision_tower.requires_grad_(False)
        
        if self.subset == "train":
            self.image_processor = transforms.Compose([
                    transforms.Normalize(mean=0.0, std=255.0),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Resize((self.image_crop_size, self.image_crop_size)),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std)
                ])
        else:
            self.image_processor = transforms.Compose([
                transforms.Normalize(mean=0.0, std=255.0),
                transforms.Resize((self.image_crop_size, self.image_crop_size)),
                transforms.Normalize(mean=self.image_mean, std=self.image_std)
            ])

        self.is_loaded = True
        
    @torch.no_grad
    def forward(self, videos):
        videos = einops.rearrange(videos, 'b t c h w -> b c t h w')
        # video: [B, C, T, H, W]
        B, C, T, H, W = videos.shape
        outputs = []

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk = videos[:, :, start:end]   # [B, C, chunk_len, H, W]
            chunk_len = chunk.shape[2]

            # pad if not enough frames
            if chunk_len < self.chunk_size:
                pad_len = self.chunk_size - chunk_len
                pad = torch.zeros(B, C, pad_len, H, W, device=videos.device, dtype=videos.dtype)
                chunk = torch.cat([chunk, pad], dim=2)  # time dim is 2

            # MARLIN forward
            out = self.vision_tower(chunk)  # [B, chunk_size, D]
            out = out[:, :chunk_len]  # remove padded part

            outputs.append(out)

        video_features = torch.cat(outputs, dim=1)  # [B, T, D]
            
        return video_features
            
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.encoder_embed_dim

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
    

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
            return self.avhubert_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    
# Audio Encoder
class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)

    def load_model(self):
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.audio_tower_name)

        self.audio_tower = WhisperForConditionalGeneration.from_pretrained(self.audio_tower_name).model.encoder
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_feature = self.audio_tower(audio.to(device=self.device, dtype=self.dtype).unsqueeze(0)).last_hidden_state
                audio_features.append(audio_feature)
        else:
            audio_features = self.audio_tower(audios.to(device=self.device, dtype=self.dtype)).last_hidden_state

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.d_model

    @property
    def num_mel_bins(self):
        return self.config.num_mel_bins
    
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
        self.audio_processor = AutoFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)
        self.is_loaded = True
        
    
    
    @torch.no_grad
    def forward(self, audio):
        logits = self.audio_tower(audio)
        return logits.last_hidden_state
    
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


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if  'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip2' in vision_tower:
        vision_tower = Siglip2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'AV-HuBERT' in vision_tower:
        vision_tower = AVHubert(vision_tower, **kwargs)
    elif 'marlin' in vision_tower:
        vision_tower = MARLINVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    return vision_tower
    
    

def build_audio_tower(audio_tower_cfg, delay_load=False, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    if "BEATs" in audio_tower:
        if not delay_load:
            beats_checkpoint = torch.load(audio_tower, map_location='cpu')
            if 'cfg' in beats_checkpoint:
                beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
            else:
                beats_cfg = BEATsConfig()
            beats = BEATs(beats_cfg)
            if not audio_tower.endswith('.bin'):
                print(beats.load_state_dict(beats_checkpoint['model']))
            else:
                filtered_checkpoint = {}
                prefix = 'model.audio_tower.'
                for key, value in beats_checkpoint.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]  # 去除前缀
                        filtered_checkpoint[new_key] = value
                print(f"Load audio tower from pretrain:\n{audio_tower}")
                print(beats.load_state_dict(filtered_checkpoint, strict=False))
        else:
            print("Load audio tower not from pretrain")
            beats_cfg = BEATsConfig()
            beats = BEATs(beats_cfg)
        
        audio_tower = beats.to(torch.float32)
        audio_tower_cfg = beats_cfg
            
    elif "whisper" in audio_tower:
        whisper = WhisperAudioTower(audio_tower=audio_tower, delay_load=delay_load)
        whisper_cfg = whisper.config
        
        audio_tower = whisper
        audio_tower_cfg = whisper_cfg
    
    elif 'hubert' in audio_tower:
        hubert = Hubert(audio_tower=audio_tower, delay_load=delay_load)
        hubert_cfg = hubert.config
        
        audio_tower = hubert
        audio_tower_cfg = hubert_cfg
        
    else:
        raise ValueError(f'Unknown audio tower: {audio_tower}')
    
    return audio_tower, audio_tower_cfg


def build_audio_tokenizer(audio_tower_cfg, delay_load=False, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tokenizer', getattr(audio_tower_cfg, 'audio_tokenizer', None))
    if not delay_load:
        beats_checkpoint = torch.load(audio_tower, map_location='cpu')
        if 'cfg' in beats_checkpoint:
            beats_tokenizer_cfg = TokenizersConfig(beats_checkpoint['cfg'])
        else:
            beats_tokenizer_cfg = TokenizersConfig()
        beats_tokenizer = Tokenizers(beats_tokenizer_cfg)
        print(beats_tokenizer.load_state_dict(beats_checkpoint['model']))
    else:
        print("Load audio tower not from pretrain")
        beats_tokenizer_cfg = TokenizersConfig()
        beats_tokenizer = Tokenizers(beats_tokenizer_cfg)
    return beats_tokenizer, beats_tokenizer_cfg


def build_qformer(qformer_cfg):
    qformer_config = BertConfig.from_pretrained(qformer_cfg.qformer_model)
    qformer_config.num_hidden_layers = qformer_cfg.qformer_layers
    qformer_config.encoder_width = qformer_cfg.embed
    qformer_config.hidden_size = qformer_cfg.qformer_dim 
    qformer_config.add_cross_attention = True
    qformer_config.cross_attention_freq = 1
    qformer_config.query_length = qformer_cfg.max_queries
    qformer = BertLMHeadModel(qformer_config)
        
    return qformer, qformer_config

def init_video_Qformer(qformer_cfg, causal_encoder=False, cache_dir=""):
    encoder_config = BertConfig.from_pretrained(qformer_cfg.qformer_model, cache_dir=cache_dir)
    if qformer_cfg.qformer_layers > 0:
        encoder_config.num_hidden_layers = qformer_cfg.qformer_layers
        encoder_config.cross_attention_freq = 1
        encoder_config.causal_encoder = causal_encoder
    else:
        encoder_config.cross_attention_freq = 2
    encoder_config.encoder_width = qformer_cfg.embed
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.query_length = qformer_cfg.max_queries
    encoder_config.use_cache = False
    # encoder_config.gradient_checkpointing = True
    encoder_config.gradient_checkpointing = False
    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, qformer_cfg.max_queries, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, encoder_config, query_tokens

def build_sr_predictor(sr_cfg, **kwards):
    args = SpeechRatePredictorConfig()
    args.num_layers = sr_cfg.sr_predictor_layers
    sr_predictor = Speech_Rate_Predictor(config=args, checkpoint_path=sr_cfg.sr_predictor, **kwards)
    return sr_predictor