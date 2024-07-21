import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model.DataSet import CO_Set
from model.BertConfigConv import BertConfig
from model.CO_Bert import BertForPretrain
from torch.utils.data import Dataset, DataLoader,Subset
from model.Trainer import ModelTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
mode = 'Raman'

json_file = f'config/config_conv_{mode}.json'
config = BertConfig.from_json_file(json_file)
model = BertForPretrain(config)
print(get_parameter_number(model))
# print(model)
# print(model.parameters())
test_set = CO_Set('/home/yanggk/Data/CO_Bert/washed2.txt',mode=mode)


bsz = 256

train_idx = np.load('data/train.npy')
val_idx = np.load('data/valid.npy')
test_idx = np.load('data/test.npy')

train_dataset = Subset(test_set,train_idx)
validate_dataset = Subset(test_set,val_idx)
test_dataset = Subset(test_set,test_idx)
print(f'======train size:{len(train_dataset)}======')
print(f'======valid size:{len(validate_dataset)}======')
print(f'======test size:{len(test_dataset)}======')

# train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(test_set, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=8,pin_memory=True)
validate_loader = DataLoader(validate_dataset, batch_size=bsz, shuffle=True, num_workers=8,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

trainer = ModelTrainer(model,mode)
trainer.start_training(train_dataloader=train_loader,
                       valid_dataloader=validate_loader,
                       test_loader=test_loader,
                       epochs=1000,
                       continue_train=False)

