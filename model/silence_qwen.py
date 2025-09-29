from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import  (
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3Config
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from peft import LoraConfig, get_peft_model

from .silence_model import SilenceMetaForCausalLM, SilenceMetaModel
from .submodels.modules import MMClassifierHead

class SilenceQwen3Config(Qwen3Config):
    model_type = "silence_qwen3"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "silence_qwen3"
        
class SilenceQwen3Model(SilenceMetaModel, Qwen3Model):
    config_class = SilenceQwen3Config
    
    def __init__(self, config):
        super(SilenceQwen3Model, self).__init__(config)


class SilenceQwen3ForCausalLM(Qwen3ForCausalLM, SilenceMetaForCausalLM):
    config_class = SilenceQwen3Config
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SilenceQwen3Model(config)
        # self.mlp = MMClassifierHead(config.hidden_size, config.num_classes)
        self.post_init()
        
    def get_model(self):
        return self.model
    
    # CORRECTED forward METHOD
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # The multimodal preparation logic has been removed from here.
        # This method is now a clean passthrough, compatible with the generate loop.
        
        # We can add logic to handle a `source` kwarg for training/inference outside of `generate`
        
        source = kwargs.pop("source", None)

        if past_key_values is None and inputs_embeds is None and source is not None:
            inputs_embeds, attention_mask, labels = self.prepare_inputs_embeds_for_multimodal(source, subset="train")
            input_ids = None
            
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **kwargs
        )
        
        return outputs
        
        # hidden_states = outputs.hidden_states[:, -1, :]
        # logits = self.mlp(hidden_states)
    
        # if labels is not None:
        #     shift_logits = logits[:, :-1, :].contiguous()
        #     shift_labels = labels[:, 1:].contiguous()

        #     loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(shift_logits.view(-1, self.config.num_classes),
        #                     shift_labels.view(-1))
        # else:
        #     loss = None
        
        # return CausalLMOutputWithPast(
        #         loss=loss,
        #         logits=logits,
        #         past_key_values=outputs.past_key_values,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #     ) 
        
    @torch.no_grad()
    def generate(
        self,
        source: Optional[dict] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
       
        # If a multimodal source is provided, prepare the embeddings here.
        if source is not None:
            inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds_for_multimodal(source, subset="test")
            # Pass the prepared embeddings and mask to the generation function.
            kwargs["inputs_embeds"] = inputs_embeds
            kwargs["attention_mask"] = attention_mask
            # Ensure `input_ids` is not also passed, to avoid conflict.
            kwargs.pop("input_ids", None)

        # Call the parent generate method. 
        # Crucially, we do NOT pass `source` itself, as its job is done.
        return super().generate(**kwargs)
        
        
if __name__ == "__main__":
    config = SilenceQwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    config.output_dims = 3
    config.mm_vision_tower = "openai/clip-vit-base-patch32"
    config.mm_audio_tower = "openai/whisper-medium"
    config.mm_vision_select_layer = -2
    config.use_qformer = True
    config.modality_fuse = "concat"
    config.sr_predictor = "/n/work1/muyun/Model/MMS_LLAMA/sr_predictor/checkpoint.pt"
    config.qformer_layers = 2
    config.qformer_dim = 1024
    config.queries_per_sec = 3
    config.sr_predictor_layers = 2
    config.mm_hidden_size_a = 1024
    config.mm_hidden_size = 1024
    model = SilenceQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", config=config, device_map="auto")
    
    vision_tower = model.get_vision_tower()
    audio_tower = model.get_audio_tower()
    
    print(vision_tower)
    print(audio_tower)
