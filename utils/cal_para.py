import torch
import torch.nn as nn
from model.BertConfigConv import BertConfig
from model.conv_bert import BertForPretrain



json_file = 'config/config_conv.json'
config = BertConfig.from_json_file(json_file)
model = BertForPretrain(config)
print(model)
# print(model.parameters())
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))