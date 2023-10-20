import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,multilabel_confusion_matrix
import seaborn as sns

class ModelTrainer:
    def __init__(self, model):
        self.model = model.cuda()
        # self.loss = loss_func
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,patience=50,min_lr=1e-06,verbose=True)

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for inputs, label, prop in tqdm(dataloader):
            self.optimizer.zero_grad()
            loss, _, cls_pred = self.model(inputs.cuda(),prop.cuda(),label.cuda())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, label, prop in dataloader:
                loss, _, _ = self.model(inputs.cuda(),prop.cuda(),label.cuda())
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def test(self, dataloader):
        self.model.load_state_dict(torch.load('best_model.pth'))  # PATH是你的模型文件路径
        self.model.eval()
        total_loss = 0.0
        draw_datas = []
        label_datas = []
        with torch.no_grad():
            for inputs, label, prop in dataloader:
                loss, eads_pred, lab_pred = self.model(inputs.cuda(),prop.cuda(),label.cuda())
                draw_data = (np.array(eads_pred.squeeze().cpu()),np.array(prop.squeeze().cpu()))
                label_data = (np.array(lab_pred.squeeze().cpu()),np.array(label.squeeze().cpu()))
                total_loss += loss.item()
                draw_datas.append(draw_data)
                label_datas.append(label_data)
        return total_loss / len(dataloader),draw_datas,label_datas

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
        test_loss,draw_datas,Label_data = self.test(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        # print(draw_datas)
        self.draw_pict(draw_datas)
        self.draw_hm(Label_data)

    def draw_pict(self,draw_datas):
        props_idx = [i for i in range(6)]
        for i in props_idx:
            xs = []
            ys = []
            for item in draw_datas:
                xs.append(item[0][i])
                ys.append(item[1][i])
            plt.scatter(ys,xs)
            r2 = r2_score(xs,ys)
            plt.xlabel('pred')
            plt.ylabel('true')
            plt.title(f'R2:{round(r2,4)}')
            plt.savefig(f'figs/{i}.png')
            plt.cla()

    def draw_hm(self,label_data):
        # cm = np.zeros((7,2,2))
        preds = []
        truths = []
        for i in range(len(label_data)):
            truths.append(label_data[i][1])
            preds.append((label_data[i][0] > 0.5).astype(int))
        cm = multilabel_confusion_matrix(truths,preds)
        for i in range(cm.shape[0]):
            sns.heatmap(cm[i], annot=True)
            plt.ylabel('Predicted')
            plt.xlabel('Truth')
            plt.savefig(f'figs/test{i}.png')
            plt.cla()
            plt.close()

