from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextAttention,
    Speech2TextDecoder,
    Speech2TextDecoderLayer
    )
from transformers.activations import ACT2FN
from transformers.models.speech_to_text.configuration_speech_to_text import Speech2TextConfig
from .av2text_config import AV2TextConfig
from typing import Optional
import torch.nn as nn
class AVTransformerAttention(Speech2TextAttention):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Speech2TextConfig] = None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            is_decoder,
            bias,
            is_causal,
            config
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)



class AVTransformerDecoderLayer(Speech2TextDecoderLayer):
    def __init__(self, config: Speech2TextConfig):
        super().__init__(config)
        self.embed_dim = config.decoder_hidden_size

        self.self_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            config.encoder_hidden_size,
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

SPEECH_TO_TEXT_ATTENTION_CLASSES = {"eager": AVTransformerAttention}

class AVTransformerDecoder(Speech2TextDecoder):
    def __init__(self, config: AV2TextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([AVTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])