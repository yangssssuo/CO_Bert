import torch

conv = torch.nn.Conv1d(1,2,3,1)

aaa = torch.randn([32,1,16])

bbb = conv(aaa)

print(bbb.shape)