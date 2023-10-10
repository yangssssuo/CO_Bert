import pandas as pd
import torch
import torch.nn as nn
from model.DataSet import CO_Set
from model.BertConfig import BertConfig
from model.mybert import BertForMaskedLM,loss_func
from torch.utils.data import Dataset, DataLoader


class ModelTrainer:
    def __init__(self, model):
        self.model = model.cuda()
        self.loss = loss_func
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,patience=100,min_lr=1e-06,verbose=True)

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs.cuda())
            loss = self.loss(outputs, targets.cuda())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs.cuda())
                loss = self.loss(outputs, targets.cuda())
                
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def test(self, dataloader):
        self.model.load_state_dict(torch.load('best_model.pth'))  # PATH是你的模型文件路径
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def start_training(self, train_dataloader, valid_dataloader,test_loader, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)
            self.scheduler.step(valid_loss)
            save_log = ''
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                save_log = ', save best model!'
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}{save_log}')
        test_loss = self.test(test_loader)
        print(f'Test Loss: {test_loss:.4f}')