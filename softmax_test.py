import torch.nn as nn
import torch
import torch.nn.functional as F


aaa = torch.randn(3,1,7)
print(aaa)
bbb = F.softmax(aaa[:,:,:2],dim=2)
# print(bbb)
ccc = F.softmax(aaa[:,:,2:4],dim=2)
ddd = F.softmax(aaa[:,:,4:],dim=2)

xxx = torch.concat([bbb,ccc,ddd],dim=2)
print(xxx)