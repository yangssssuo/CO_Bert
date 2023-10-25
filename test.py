import pandas as pd
import torch
import torch.nn as nn
from model.DataSet import CO_Set
from model.BertConfig import BertConfig
from model.SingleBert import BertForPretrain
from torch.utils.data import Dataset, DataLoader
from model.Trainer import ModelTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

json_file = 'config/config_test.json'
config = BertConfig.from_json_file(json_file)
model = BertForPretrain(config)
# print(model)
# print(model.parameters())
test_set = CO_Set('/home/yanggk/Data/CO_Bert/washed.txt',mode='Raman')

# for item,_,_ in test_set:
#     print(item.max())


set_size = len(test_set)
# print(set_size)
train_size = int(0.9*set_size)
test_size = int(0.05*set_size)
val_size = set_size - train_size -test_size


bsz = 512


train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(test_set, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=24)
validate_loader = DataLoader(validate_dataset, batch_size=bsz, shuffle=True, num_workers=24)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=24)

trainer = ModelTrainer(model)
trainer.start_training(train_dataloader=train_loader,valid_dataloader=validate_loader,test_loader=test_loader,epochs=500)

# count = 0
# for n,(input,_,_) in enumerate(train_loader):
#     # print(input.max())
#     for item in input:
#         # print(item.max())
#         if item.max() != 100.0:
#             count +=1
# print(count)