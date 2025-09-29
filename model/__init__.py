from .silence_qwen import SilenceQwen3ForCausalLM, SilenceQwen3Config
from .silence_llama import SilenceLlama3ForCausalLM, SilenceLlama3Config
from .silence_mlp import SilenceMLPForCausalLM, SilenceMLPConfig
from .silence_model import SilenceMetaModel, SilenceMetaForCausalLM
from .silence_perceiver import SilencePerceiverConfig, SilencePerceiverForCausalLM

VLLMs = {
    "SilenceQwen3" : SilenceQwen3ForCausalLM,
    "SilenceLlama3" : SilenceLlama3ForCausalLM,
    "SilenceMLP": SilenceMLPForCausalLM,
    "SilencePerceiver": SilencePerceiverForCausalLM
}

VLLMConfigs = {
    "SilenceQwen3" : SilenceQwen3Config,
    "SilenceLlama3" : SilenceLlama3Config,
    "SilenceMLP": SilenceMLPConfig,
    "SilencePerceiver": SilencePerceiverConfig
}