import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

class CO_Set(Dataset):
    '''
    length 30 
    [atom info 6 + freq*6 + ir*6 + raman*6 + eads +delta e + eb + d band + wf + dipole alpha]
    '''
    def __init__(self,name_lis,mode='IR') -> None:
        super(CO_Set,self).__init__()
        self.file_path = '/home/yanggk/Data/CO_Bert/'
        aaa = pd.read_csv(name_lis,header=None)
        self.name = aaa.values.reshape(-1).tolist()
        self.mode = mode

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        name = self.name[index]

        if self.mode == 'IR':
            path = self.file_path + 'IR/data/' + name + '.txt'
        elif self.mode == 'Raman':
            path = self.file_path + 'Raman/data/' + name + '.txt'
        data = pd.read_csv(path,header=None,index_col=0,dtype='float32').values
        # ori_spec = self.min_max(torch.tensor(data))
        ori_spec = torch.tensor(data)
        # ori_spec = torch.nn.functional.sigmoid(torch.tensor(data))
        
        prop_path = self.file_path + 'Property/' + name + '.csv'
        prop_data = pd.read_csv(prop_path,index_col=0).values.tolist()
        prop_data = torch.tensor(prop_data)

        label = self._get_label(name)


        return ori_spec,label,prop_data
    
    def min_max(self,t):
        min_t = torch.min(t)
        max_t = torch.max(t)
        return (t - min_t) * 100 / (max_t-min_t)
    
    def _get_label(self,name):
        name_lis = name.split('-')
        sys = name_lis[0]
        metal = name_lis[1]
        pos = name_lis[2]
        sys_tok = torch.tensor([[1,0]]) if sys == 'CO' else torch.tensor([[0,1]])
        metal_tok = torch.tensor([[1,0]]) if metal == 'Au' else torch.tensor([[0,1]])
        if pos == 'top':
            pos_tok = torch.tensor([[1,0,0]])
        elif pos == 'hollow':
            pos_tok = torch.tensor([[0,1,0]])
        elif pos == 'bridge':
            pos_tok = torch.tensor([[0,0,1]])
        label = torch.concat([sys_tok,metal_tok,pos_tok],dim=1)
        return label

if __name__ == '__main__':
    aaa = CO_Set('/home/yanggk/Data/CO_Bert/name.txt')
    # print(aaa[0])
    # print(aaa[1][0].shape)
    print(aaa[1][1])
    print(aaa[10000][1])

    # aaa = pd.read_csv("/home/yanggk/Data/CO_Bert/Property/CO-Ag-bridge-1-1.csv",index_col=0)
    # bbb = aaa.values.tolist() 
    # bbb = torch.tensor(bbb)
    # print(bbb.t().shape)

