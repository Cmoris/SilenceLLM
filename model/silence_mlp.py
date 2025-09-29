from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import  (
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutput

from .silence_model import SilenceMetaForCausalLM, SilenceMetaModel
from .submodels.perceiver import PerceiverLM

class SilenceMLPConfig(PretrainedConfig):
    model_type = "silence_mlp"
    
    def __init__(self, 
                 vocab_size = 151936, 
                 hidden_size=1024,
                 num_classes=2,
                 depth = 6,                   # depth of net
                 max_seq_len = 2048,          # maximum sequence length
                 num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
                 latent_dim = 512,            # latent dimension
                 cross_heads = 1,             # number of heads for cross attention. paper said 1
                 latent_heads = 8,            # number of heads for latent self attention, 8
                 cross_dim_head = 64,         # number of dimensions per cross attention head
                 latent_dim_head = 64,        # number of dimensions per latent self attention head
                 weight_tie_layers = False,    # whether to weight tie layers (optional, as indicated in the diagram)
                 **kwargs):
        super().__init__(**kwargs)
        self.model_type = "silence_mlp"
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len         
        self.num_latents = num_latents
        self.latent_dim = latent_dim           
        self.cross_heads = cross_heads             
        self.latent_heads = latent_heads            
        self.cross_dim_head = cross_dim_head        
        self.latent_dim_head = latent_dim_head        
        self.weight_tie_layers = weight_tie_layers   

        
class SilenceMLPModel(SilenceMetaModel, PreTrainedModel):
    config_class = SilenceMLPConfig
    
    def __init__(self, config):
        super(SilenceMLPModel, self).__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class SilenceMLPForCausalLM(PreTrainedModel, SilenceMetaForCausalLM):
    config_class = SilenceMLPConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SilenceMLPModel(config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self._dynamic_tied_weights_keys = set([
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.Qformer.cls.predictions.decoder.weight",
            "model.Qformer.bert.embeddings.word_embeddings.weight",
            "model.Qformer.cls.predictions.bias",
            "model.Qformer.cls.predictions.decoder.bias"
        ])
        self.post_init()
        
    def get_model(self):
        return self.model
    
    def forward(self, source) -> CausalLMOutput:
        labels = source["classes"]
        
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        
        has_audio = 'audio' in source and source['audio'] is not None
        has_video = 'video' in source and source['video'] is not None
        
        with torch.no_grad():
            audio_enc_out = None
            video_enc_out = None
            
            if has_audio:
                audio_enc_out = audio_tower(source["audio"])
            
            if has_video:
                if "AV-HuBERT" in self.config.mm_vision_tower:
                    video_enc_out, padding_mask = vision_tower(source)
                elif "marlin" in self.config.mm_vision_tower:
                    video_enc_out = vision_tower(source['video'])
                else:
                    frames = list(source['video'].unbind(1))
                    video_enc_out = vision_tower(frames)

        if has_video:
            feat_lengths = torch.sum(source['video_mask'], dim=1).tolist()
            max_feat_len = max(feat_lengths)
        elif has_audio:
            feat_lengths = [int(audio_enc_out.size(1)/2)] * audio_enc_out.size(0)
            max_feat_len = max(feat_lengths)
        
        # ============================
        # 2. Speech rate predictor and query length calculation
        # ============================
        if getattr(self.config, "use_sr_predictor", False) and has_audio:
            len_queries, resized_len_list = self.query_length_calculation(audio_enc_out, feat_lengths, max_feat_len)
        elif has_video:
            len_queries = [max(int(vid_len / 30 * self.config.queries_per_sec), self.config.queries_per_sec) for vid_len in feat_lengths]
            resized_len_list = feat_lengths
        elif has_audio:
            len_queries = [max(int(aud_len / 25 * self.config.queries_per_sec), self.config.queries_per_sec) for aud_len in feat_lengths]
            resized_len_list = feat_lengths
        else:
            raise ValueError("At least one modality must be provided.")
        
        # ============================
        # 3. Feature processing and modality fusion
        # ============================
        if has_audio:
            audio_enc_out = self.get_model().mm_projector_a(audio_enc_out.transpose(1, 2)).transpose(1, 2) # Process Whisper features with 1D conv: (B x T x D) -> (B x T x D')
        
        if has_video:
            if self.config.use_qformer:
                video_enc_out = self.get_model().mm_projector_v(video_enc_out)
                len_feat = feat_lengths 
            else:
                # Without Qformer: downsample the visual feature and corresponding padding mask (e.g., 25Hz -> 12.5Hz)
                padding_mask = source['video'].ne(0)[:, 1::2]
                padding_mask = (~padding_mask).long()
                len_feat = torch.sum(padding_mask, dim=1).tolist()
                video_enc_out = self.get_model().mm_projector_v(
                    video_enc_out
                )
        elif has_audio:
            len_feat = feat_lengths 
        
        if has_video and has_audio:
            if self.config.modality_fuse == 'cross-att':
                pass
            else:
                B1, T_v, D1 = video_enc_out.size()
                B2, T_a, D2 = audio_enc_out.size()
                assert B1 == B2, "Batch size should be same"
                
                if T_a > T_v:
                    audio_enc_out = audio_enc_out[:, :T_v, :]
                else:
                    pad_len = T_v - T_a
                    audio_enc_out = F.pad(audio_enc_out, (0, 0, 0,pad_len))
        elif has_audio:
            video_enc_out = torch.zeros_like(audio_enc_out)
        elif has_video:
            audio_enc_out = torch.zeros_like(video_enc_out)
        
        # Fuse modalities based on configuration
        if self.config.modality_fuse == 'concat':
            av_feat = torch.cat([audio_enc_out, video_enc_out], dim=2)
        elif self.config.modality_fuse == 'add':
            av_feat = audio_enc_out + video_enc_out
        elif self.config.modality_fuse == 'cross-attn':
            av_feat = self.get_model().multimodal_attention_layer(
                audio_feature=audio_enc_out,
                visual_feature=video_enc_out
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.config.modality_fuse}")
        
        
        if self.config.use_qformer and av_feat is not None:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            query_output = self.get_model().mm_projector(query_output)
        else:
            # Directly map fused AV features to LLM embedding space
            query_output = self.get_model().mm_projector(av_feat)
        
        hidden_states = self.fc(query_output.mean(1))
        logits = self.lm_head(query_output)
        loss = F.cross_entropy(hidden_states, labels, ignore_index=-100)
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
        
    
   