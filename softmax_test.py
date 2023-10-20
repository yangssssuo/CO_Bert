import torch.nn as nn
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix

# 假设我们有以下测试标签和预测标签
# 每个样本有三个标签，每个标签有两个类别
y_true = [[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]
y_pred = [[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]]
print(y_pred)
# 计算多标签混淆矩阵
cm = multilabel_confusion_matrix(y_true, y_pred)

# 打印多标签混淆矩阵
print(cm.shape)
for i in range(cm.shape[0]):
    sns.heatmap(cm[i], annot=True)
    plt.savefig(f'figs/test{i}.png')
    plt.cla()
    plt.close()
