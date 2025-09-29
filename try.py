# import torch
# from transformers import AutoModel, AutoConfig

# model = AutoModel.from_pretrained(
#     "ControlNet/marlin_vit_base_ytf",  # or other variants
#     trust_remote_code=True
# )
# config = AutoConfig.from_pretrained(
#     "ControlNet/marlin_vit_base_ytf",
#     trust_remote_code=True
# )
# print(config)
# tensor = torch.rand([4, 3, 16, 224, 224])  # (B, C, T, H, W)
# output = model(tensor)  # torch.Size([1, 1568, 384])
# print(output.size())

# # import evaluate

# # try:
# #     wer_metric = evaluate.load("wer")
# #     print("✅ 成功加载 WER 指标！")

# #     # 做一个简单的计算测试
# #     predictions = ["this is a test"]
# #     references = ["this is the test"]
# #     wer_score = wer_metric.compute(predictions=predictions, references=references)
# #     print(f"计算出的 WER: {wer_score}")

# # except Exception as e:
# #     print(f"❌ 加载或计算 WER 时出错: {e}")

from transformers import HubertModel, HubertConfig, AutoProcessor, LlamaForCausalLM
import torch
import torch.nn as nn

model = LlamaForCausalLM.from_pretrained("/n/work1/muyun/Model/Llama3.2-3B-Instruct-hf")
print(model)