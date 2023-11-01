from sklearn.ensemble import ExtraTreesRegressor
from model.DataSet import CO_Set
from torch.utils.data import Dataset, DataLoader,Subset
import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

test_set = CO_Set('/home/yanggk/Data/CO_Bert/washed.txt',mode='Raman')

reg = ExtraTreesRegressor(n_estimators=100, random_state=0,n_jobs=24)

bsz = len(test_set)

bsz = 512

train_idx = np.load('data/train.npy')
val_idx = np.load('data/valid.npy')
test_idx = np.load('data/test.npy')

train_dataset = Subset(test_set,train_idx)
validate_dataset = Subset(test_set,val_idx)
test_dataset = Subset(test_set,test_idx)

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=24)
validate_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=True, num_workers=24)
spec_train = []
prop_train = []

spec_test = []
prop_test = []

for n,(inputs, label, prop) in enumerate(train_loader):
    input_np = inputs.squeeze()
    # print(input_np.shape)
    spec_train.append(input_np)

    # print(input_np)
    prop_np = prop.squeeze()
    prop_train.append(prop_np)
    # print(prop_np.shape)
    # reg.fit(input_np,prop_np)


for n,(inputs, label, prop) in enumerate(validate_loader):
    input_np = inputs.squeeze()
    # print(input_np.shape)
    spec_test.append(input_np)

    # print(input_np)
    prop_np = prop.squeeze()
    prop_test.append(prop_np)

# print(spec_train[-1].shape)

train_x = torch.concat(spec_train,dim=0).numpy()
train_y = torch.concat(prop_train,dim=0).numpy()
print(train_x.shape)


test_x = torch.concat(spec_test,dim=0).numpy()
test_y = torch.concat(prop_test,dim=0).numpy()
print(test_y[0])

reg.fit(train_x,train_y)

pred_x = reg.predict(test_x)
# print(pred_x)

r2 = r2_score(pred_x,test_y)
print(r2)

def draw_pict(draw_datas):
    props_idx = [i for i in range(6)]
    for i in props_idx:
        xs = []
        ys = []
        for n in range(len(draw_datas[0])):
            xs.append(draw_datas[0][n][i])
            ys.append(draw_datas[1][n][i])
        plt.scatter(ys,xs)
        r2 = r2_score(xs,ys)
        plt.xlabel('pred')
        plt.ylabel('true')
        plt.title(f'R2:{round(r2,4)}')
        plt.savefig(f'figs/etr{i}.png')
        plt.cla()
draw_data = (pred_x,test_y)

draw_pict(draw_data)