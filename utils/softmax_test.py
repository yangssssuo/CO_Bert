import torch.nn as nn
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix

# # 假设我们有以下测试标签和预测标签
# # 每个样本有三个标签，每个标签有两个类别
# y_true = [[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]
# y_pred = [[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]]
# print(y_pred)
# # 计算多标签混淆矩阵
# cm = multilabel_confusion_matrix(y_true, y_pred)

# # 打印多标签混淆矩阵
# print(cm.shape)
# for i in range(cm.shape[0]):
#     sns.heatmap(cm[i], annot=True)
#     plt.savefig(f'figs/test{i}.png')
#     plt.cla()
#     plt.close()

# loss_func = nn.CrossEntropyLoss()

# # 输入数据
# input_data = torch.tensor([
#     [[0.8,0.7, 0.5, 0.2, 0.5], [0.2, 0.9, 0.7,0.3, 0.2], [0.4, 0.3, 0.7,0.7, 0.1], [0.1, 0.2, 0.7,0.4, 0.8]],
#     [[0.8, 0.7,0.5, 0.2, 0.5], [0.2, 0.9, 0.7,0.3, 0.2], [0.4, 0.3, 0.7,0.7, 0.1], [0.1, 0.2, 0.7,0.4, 0.8]]
# ])

# # 目标数据
# target_data = torch.tensor([
#     [1, 1, 2, 2,2],
#     [1, 1, 2, 3,1]
# ])

# # 计算损失
# loss = loss_func(input_data, target_data)

# print(input_data.shape)
# print(target_data.shape)
# print(loss.shape)

# 定义损失函数
loss_func = nn.BCEWithLogitsLoss()

# 输入数据
input_data = torch.randn(3, 7) # 假设有10个样本，每个样本有7个预测值

# 目标数据
target_data = torch.empty(3, 7).random_(2) # 假设有10个样本，每个样本有7个真实标签

# 计算损失
loss = loss_func(input_data, target_data)

print(input_data.shape)#3,7
print(target_data.shape)#3,7
print(loss.shape)
print(input_data)
print(target_data)
print(loss)
