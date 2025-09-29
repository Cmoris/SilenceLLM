from abc import ABC, abstractmethod
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from .submodels.modules import (
    build_vision_tower, 
    build_audio_tower, 
    build_audio_tokenizer,
    build_qformer, 
    build_sr_predictor, 
    Projector, 
    Multimodal_Attention)
from .submodels.projector import load_mm_projector, build_vision_projector, build_audio_projector

class SilenceMetaModel:
    def __init__(self, config):
        super(SilenceMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector_v = build_vision_projector(config)

        if hasattr(config, "mm_audio_tower"):
            self.audio_tower, self.audio_tower_cfg = build_audio_tower(config, delay_load=True)
            self.mm_projector_a = build_audio_projector(config)
            
        if hasattr(config, "mm_qformer_model") and getattr(config, "use_qformer", True):
            self.Qformer, self.qformer_config = build_qformer(config)
            self.mm_projector = Projector(input_dim=config.qformer_dim,
                                        hidden_dim=math.floor((config.qformer_dim + config.hidden_size) / 2),
                                        output_dim=config.hidden_size)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.max_queries, config.qformer_hidden_size, requires_grad=True)
            )
            self.ln_head = nn.LayerNorm(config.embed)
            
            if getattr(config, "modality_fuse", None) == "cross-attn":
                self.multimodal_attention_layer = Multimodal_Attention(qdim=config.mm_hidden_size_v,
                                                                        kdim=config.mm_hidden_size_a, num_heads=8)
        else:
            if hasattr(config, "embed"): 
                self.mm_projector = Projector(input_dim=config.embed,
                                            hidden_dim=math.floor((config.embed + config.hidden_size) / 2),
                                            output_dim=config.hidden_size)
            
        if hasattr(config, "mm_sr_predictor"):
            self.sr_predictor = build_sr_predictor(config, delay_load=True) 
        
        
            
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower
    
    def get_sr_predictor(self):
        sr_predictor = getattr(self, 'sr_predictor', None)
        return sr_predictor

    def get_Qformer(self):
        qformer = getattr(self, "Qformer", None)
        return qformer
    
    def initialize_qformer(self, model_args, fsdp=None):
        if getattr(model_args, "use_sr_predictor", True):
            self.config.use_sr_predictor = model_args.use_sr_predictor
            self.config.max_queries = int(model_args.queries_per_sec * 20 * 2)
        else:
            self.config.use_sr_predictor = model_args.use_sr_predictor
            self.config.max_queries = int(model_args.queries_per_sec * 30)
        
        if model_args.modality_fuse == 'concat':
            if self.config.mm_hidden_size_v == 0:
                self.config.mm_hidden_size_v = self.config.mm_hidden_size_a
            elif self.config.mm_hidden_size_a == 0:
                self.config.mm_hidden_size_a = self.config.mm_hidden_size_v
            self.config.embed = self.config.mm_hidden_size_a + self.config.mm_hidden_size_v
        elif model_args.modality_fuse == 'add':
            self.config.embed = self.config.mm_hidden_size_a
        elif model_args.modality_fuse == 'cross-attn':
            self.multimodal_attention_layer = Multimodal_Attention(qdim=self.config.mm_hidden_size_v,
                                                                   kdim=self.config.mm_hidden_size_a, num_heads=8)
            self.config.embed = self.config.mm_hidden_size_v
        
        self.config.queries_per_sec = model_args.queries_per_sec
        self.config.use_qformer = True
        self.config.qformer_layers = model_args.qformer_layers
        self.config.qformer_dim = model_args.qformer_dim 
        self.config.qformer_model = model_args.qformer_model
        self.config.mm_qformer_model = model_args.qformer_model
        self.config.window_level_Qformer = model_args.window_level_Qformer
        self.config.modality_fuse = model_args.modality_fuse
        if self.config.window_level_Qformer:
            self.config.second_per_window = model_args.second_stride
            self.config.second_stride = model_args.second_stride

        if self.get_Qformer() is None:
            self.Qformer, qformer_config = build_qformer(self.config)
        else:
            qformer_config = self.qformer_config
            for p in self.Qformer.parameters():
                p.requires_grad = True
            
        self.config.qformer_hidden_size = qformer_config.hidden_size
        
        
        if getattr(self, "query_tokens", None) is None:
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.config.max_queries, qformer_config.hidden_size)
            )
        
        self.query_tokens.requires_grad = True
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        
        self.ln_head = nn.LayerNorm(self.config.embed)
        
    
    def initialize_sr_predictor(self, model_args, fsdp=None):
        if self.get_sr_predictor() is None:
            self.sr_predictor = build_sr_predictor(model_args, delay_load=False)
        else:
            self.sr_predictor.load_model()
            for p in self.sr_predictor.parameters():
                p.requires_grad = False
        
        self.config.use_sr_predictor = True
        self.config.sr_predictor_layers = model_args.sr_predictor_layers
        self.config.sr_predictor = model_args.sr_predictor
        self.config.mm_sr_predictor = model_args.sr_predictor
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        
        self.config.mm_vision_tower = vision_tower
        
        if self.get_vision_tower() is None:
            print("Load vision tower")
            vision_tower = build_vision_tower(model_args)
    
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_v_type = getattr(model_args, 'mm_projector_v_type', 'linear')
        self.config.mm_hidden_size_v = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector_v', None) is None:
            self.mm_projector_v = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector_v.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    print(f"Load mm mlp adapter:\n{pretrain_mm_mlp_adapter}")
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector_v.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector_v.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter_a
        self.config.mm_audio_tower = audio_tower
        if self.get_audio_tower() is None:
            print("Load audio tower")
            audio_tower, audio_tower_cfg = build_audio_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
                audio_tower_cfg = self.audio_tower_cfg[0]
            else:
                audio_tower = self.audio_tower
                audio_tower_cfg = self.audio_tower_cfg
                
            audio_tower.load_model()
            
        self.config.use_mm_proj = True
        self.config.mm_projector_a_type = getattr(model_args, 'mm_projector_a_type', 'linear')
        self.config.mm_hidden_size_a = audio_tower.hidden_size
        
        if getattr(self, 'mm_projector_a', None) is None:
            self.mm_projector_a = build_audio_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector_a.parameters():
                p.requires_grad = True
        if pretrain_mm_mlp_adapter is not None:
            print(f"Load mm mlp adapter-a:\n{pretrain_mm_mlp_adapter}")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector_a.load_state_dict(get_w(mm_projector_weights, 'mm_projector_a'), strict=True)
            
    def initialize_projector(self, model_args, fsdp=None):
        if getattr(self, "mm_projector", None) is None and hasattr(model_args, "qformer_model"):
            self.mm_projector = Projector(input_dim=self.config.qformer_dim ,
                                            hidden_dim=math.floor((self.config.qformer_dim + self.config.hidden_size)/2),
                                            output_dim=self.config.hidden_size)
        elif getattr(self, "mm_projector", None) is None :
            self.mm_projector = Projector(input_dim=self.config.embed,
                                        hidden_dim=math.floor((self.config.embed + self.config.hidden_size)/2),
                                        output_dim=self.config.hidden_size) 


class SilenceMetaForCausalLM(ABC):
    
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()
    
    def get_sr_predictor(self):
        return self.get_model().get_sr_predictor()
    
    def get_qformer(self):
        return self.get_model().get_Qformer()
    
    def prepare_inputs_embeds_for_multimodal(self, source, **kwargs):
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        
        has_audio = 'audio' in source and source['audio'] is not None
        has_video = 'video' in source and source['video'] is not None
        
        with torch.no_grad():
            audio_enc_out = None
            video_enc_out = None
            
            if has_audio:
                if "BEATs" in self.config.mm_audio_tower:
                    audio_enc_out = audio_tower.extract_features(source["audio"])[0]
                else:
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

    
        instructions = source['input_ids']  # List[torch.Tensor] of length B
        if kwargs['subset'] == "train":
            labels = source['labels']           # List[torch.Tensor] of length B
        elif kwargs['subset'] == 'test' or kwargs['subset'] == 'val':
            labels = None
        else:
            raise ValueError
        
        if self.config.use_qformer and av_feat is not None:

            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            
            # Map Qformer output to LLM embedding space
            query_output = self.get_model().mm_projector(query_output)
            llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_queries(
                instructions, query_output, len_queries, labels
            )
        else:
            # Directly map fused AV features to LLM embedding space
            av_feat = self.get_model().mm_projector(av_feat)
            llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_queries(
                instructions, av_feat, len_feat, labels
            )

        return llm_inputs, attention_mask, llm_labels
    
    def prepare_inputs_labels_for_queries(self, instructions, queries, len_queries, labels=None):
        llm_input_list = []
        llm_labels_list = []
        lengths = []  
        
        for i in range(len(instructions)):
            instruction = instructions[i]
            len_query = len_queries[i]
            query = queries[i][:len_query, :]

            inst_emb = self.get_model().embed_tokens(instruction.unsqueeze(0)).squeeze(0)
            if labels is not None:
                label = labels[i]
                label_emb = self.get_model().embed_tokens(label.unsqueeze(0)).squeeze(0)
                combined = torch.cat([inst_emb, query, label_emb], dim=0)
            else:
                combined = torch.cat([inst_emb, query], dim=0)

            llm_input_list.append(combined)
            lengths.append(combined.size(0)) 
            
            if labels is not None:
                label_mask = torch.full((combined.size(0),), -100, dtype=instruction.dtype, device=instruction.device)
                offset = inst_emb.size(0) + query.size(0)
                label_mask[offset:] = labels[i]
                llm_labels_list.append(label_mask)

        # Determine the maximum sequence length across the batch
        max_seq_len = max(lengths)
        batch_size = len(llm_input_list)
        embedding_dim = llm_input_list[0].size(1)

        # Prepare the pad embedding (using the provided pad token)
        pad_token_id = self.config.tokenizer_pad_token_id
        pad_token_tensor = torch.tensor([pad_token_id], device=instruction.device)
        pad_embedding = self.get_model().embed_tokens(pad_token_tensor).squeeze(0)

        # Initialize the left-padded inputs tensor with the pad embedding.
        # Each sequence will occupy the rightmost positions.
        llm_inputs = pad_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, embedding_dim).clone()
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=instruction.device)

        for i, seq in enumerate(llm_input_list):
            seq_len = seq.size(0)
            # Place the sequence at the right end, leaving pad tokens on the left
            llm_inputs[i, max_seq_len - seq_len:] = seq
            attention_mask[i, max_seq_len - seq_len:] = 1

        if labels is not None:
            llm_labels = torch.full((batch_size, max_seq_len), -100, dtype=instruction.dtype, device=instruction.device)
            for i, lab in enumerate(llm_labels_list):
                lab_len = lab.size(0)
                llm_labels[i, max_seq_len - lab_len:] = lab
        else:
            llm_labels = None

        return llm_inputs, attention_mask, llm_labels
    
    def query_length_calculation(self, audio_enc_out, video_lengths, max_vid_len):
        sr_predictor = self.get_sr_predictor()
        
        with torch.no_grad():
            sr_predictions = sr_predictor(audio_enc_out[:,:2*max_vid_len,:][:,::4,:])
        len_queries = []
        resized_len_list = []
        for i, vid_len in enumerate(video_lengths):
            base_queries = vid_len / 30 * self.config.queries_per_sec
            factor = sr_predictions[i].item()
            # If predicted speech rate is out of acceptable range, use factor 1.0
            if factor < 1: 
                factor = 1
            elif factor > 2:
                factor = 2
            adjusted_queries = int(base_queries * factor)
            query_count = max(adjusted_queries, self.config.queries_per_sec)
            len_queries.append(query_count)
            resized_len_list.append(factor*vid_len) # resized av feat

        return len_queries, resized_len_list

    def compression_using_qformer(self, len_queries, resized_len_list, len_feat, av_feat):
        Qformer = self.get_qformer()
        
        max_length = max(len_queries)
        B, T, C = av_feat.size()
        
        resized_av_feats = torch.zeros(B,int(max(resized_len_list)), av_feat.size(2)).to(av_feat.device).to(av_feat.dtype)
        resized_padding_masks=torch.zeros(B,int(max(resized_len_list))).to(av_feat.device).to(av_feat.dtype)
        # Resize av_feat depend on the factor_list

        for bs,len_feat_bs in enumerate(len_feat): 
            new_av_feat=av_feat[bs][:len_feat_bs].transpose(0, 1).unsqueeze(0) # 1 x D x T
            resized_av_feat = F.interpolate(new_av_feat, size=int(resized_len_list[bs]), mode='linear')
            resized_av_feat=resized_av_feat.squeeze(0).transpose(0,1)
            resized_av_feats[bs, :resized_av_feat.size(0)]=resized_av_feat
            resized_padding_masks[bs, :int(resized_len_list[bs])]=1
            
        av_feat = resized_av_feats
        av_feat_atts = resized_padding_masks.long()
        av_feat = self.get_model().ln_head(av_feat)
        
        if self.config.window_level_Qformer:
            kernel = round(1500 * self.config.second_per_window / 30.0)
            stride = round(1500 * self.config.second_stride / 30.0)
            kernel = (1, kernel)
            stride = (1, stride)
            av_feat_tr = av_feat.transpose(1, 2).unsqueeze(2)
            av_feat_overlap = F.unfold(av_feat_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = av_feat_overlap.size()
            av_feat_overlap = av_feat_overlap.view(B, -1, kernel[1], L)
            av_feat_overlap = torch.permute(av_feat_overlap, [0, 3, 2, 1])
            av_feat = av_feat_overlap.reshape(-1, kernel[1], C)
            av_feat_atts = torch.ones(av_feat.size()[:-1], dtype=torch.long, device=av_feat.device)
            
        # Expand and slice query tokens: (B x max_length x token_dim)
        query_tokens = self.get_model().query_tokens.expand(av_feat.size(0), -1, -1)[:, :max_length, :]
        # Create attention mask for query tokens: (B x max_length)
        query_attn_mask = torch.zeros(av_feat.size(0), max_length, dtype=torch.long, device=av_feat.device)
        for i, qlen in enumerate(len_queries):
            query_attn_mask[i, :qlen] = 1
        
        # Run Qformer (using its BERT) with cross attention to AV features
        query_output = Qformer.bert(
            query_embeds=query_tokens,
            attention_mask=query_attn_mask,
            encoder_hidden_states=av_feat,
            encoder_attention_mask=av_feat_atts,
            return_dict=True
        )['last_hidden_state']

        query_output = query_output.view(B, -1, self.config.qformer_dim)
    
        return query_output
    