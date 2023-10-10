import pandas as pd
import torch
import torch.nn as nn
from model.DataSet import CO_Set
from model.BertConfig import BertConfig
from model.mybert import BertForMaskedLM,loss_func
from torch.utils.data import Dataset, DataLoader
from model.Trainer import ModelTrainer


json_file = 'config.json'
config = BertConfig.from_json_file(json_file)
model = BertForMaskedLM(config)
test_set = CO_Set('/home/yanggk/Data/CO_Bert/data.csv')

set_size = len(test_set)
train_size = int(0.9*set_size)
test_size = int(0.05*set_size)
val_size = set_size - train_size -test_size


bsz = 2048


train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(test_set, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=24)
validate_loader = DataLoader(validate_dataset, batch_size=bsz, shuffle=True, num_workers=24)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=24)

trainer = ModelTrainer(model)
trainer.start_training(train_dataloader=train_loader,valid_dataloader=validate_loader,test_loader=test_loader,epochs=1000)