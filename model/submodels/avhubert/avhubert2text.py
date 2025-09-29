from torch import nn
from transformers import Speech2TextModel, Speech2TextForConditionalGeneration
from .avhubert import AVHubertModel
from .av_transformer_decoder import AVTransformerDecoder
from .av2text_config import AV2TextConfig


class AV2TextModel(Speech2TextModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AVHubertModel(config)
        self.decoder = AVTransformerDecoder(config)
        
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight

class AV2TextForConditionalGeneration(Speech2TextForConditionalGeneration):
    config_class = AV2TextConfig
    def __init__(self, config):
        super().__init__(config)        
        self.model = AV2TextModel(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight