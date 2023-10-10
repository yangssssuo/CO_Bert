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
    def __init__(self,file_path) -> None:
        super(CO_Set,self).__init__()

        aaa = pd.read_csv(file_path).dropna()
        self.name = aaa.iloc[:,0:1].values
        self.sys_feature = aaa.iloc[:,1:].values

    def __len__(self):
        return len(self.sys_feature)
    
    def __getitem__(self, index):
        if self.name[index] == 'NO-Au':
            name = [8,16,7,14,79,196]
        elif self.name[index] == 'NO-Ag':
            name = [8,16,7,14,47,108]
        elif self.name[index] == 'CO-Ag':
            name = [8,16,6,12,47,108]
        elif self.name[index] == 'CO-Au':
            name = [8,16,6,12,79,196]
        name = torch.tensor(name,dtype=torch.float32)
        feature = torch.tensor(self.sys_feature[index].astype('float32'))
        ori = torch.concat((name,feature))
        masked = self._mask(ori)
        return masked.unsqueeze(1), ori.unsqueeze(1)
    

    def _mask(self,item):

        mask_idx = random.sample(range(0,30),6)
        for idx in mask_idx:
            if random.random() < 0.8:
                item[idx] = 0.0
            else:
                if random.random() < 0.5:
                    item[idx] = item[random.randint(0,29)]
                else:pass
        return item



if __name__ == '__main__':
    aaa = CO_Set('/home/yanggk/Data/CO_Bert/data.csv')
    print(aaa[0])
    print(aaa[0].shape)

