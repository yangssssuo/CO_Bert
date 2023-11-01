import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model.DataSet import CO_Set
from model.BertConfigConv import BertConfig
from model.conv_bert import BertForPretrain
from torch.utils.data import Dataset, DataLoader,Subset
from model.Trainer import ModelTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

json_file = 'config/config_conv.json'
config = BertConfig.from_json_file(json_file)
model = BertForPretrain(config)
# print(model)
# print(model.parameters())
test_set = CO_Set('/home/yanggk/Data/CO_Bert/washed.txt',mode='IR')

# for item,_,_ in test_set:
#     print(item.max())


# set_size = len(test_set)

# idx = np.arange(0,set_size)
# np.random.shuffle(idx)


# # print(set_size)
# train_size = int(0.9*set_size)
# test_size = int(0.05*set_size)
# val_size = set_size - train_size -test_size


# train_idx = idx[:train_size]
# test_idx = idx[train_size:test_size+train_size]
# val_idx = idx[test_size+train_size:]

# np.save('data/train.npy',train_idx)
# np.save('data/test.npy',test_idx)
# np.save('data/valis.npy',val_idx)

bsz = 1024

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

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=24,pin_memory=True)
validate_loader = DataLoader(validate_dataset, batch_size=bsz, shuffle=True, num_workers=24,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=24)

trainer = ModelTrainer(model,'IR')
trainer.start_training(train_dataloader=train_loader,valid_dataloader=validate_loader,test_loader=test_loader,epochs=50,continue_train=False)

