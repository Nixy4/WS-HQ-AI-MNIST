from logger import log
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, dataloader

#!Hyperparameters
'''超参数:需要在训练前设置的参数'''
EPOCH = 3 # 纪元: 训练的轮数
LEARNING_RATE = 0.001 # 学习率: 控制模型权重更新的步长
BATCH =10 # 批量大小: 每次训练使用的样本数量

#!加载数据集
img_transform = v2.Compose([
  v2.ToImage(), # 保留Alpha通道
  v2.ToDtype(torch.float32, scale=True) # 将像素值转换为浮点数并缩放到[0, 1]范围
])

train_dataset = datasets('./', transform=img_transform,train=True, download=True)
#把数据转换为Pytorch训练所需要的特定格式

#!定义一个全连接神经网络
from E6_FCNN import MyFCNN
#创建MyFCNN类的实例
my_fcnn_module = MyFCNN()
#设置优化器
optimizer = optim.Adam(my_fcnn_module.parameters(), lr=LEARNING_RATE)
#设置损失函数
loss_fn = nn.NLLLoss()
'''
NLLLoss 是 <Negative Log Likelihood Loss> 的缩写，直译为“负对数似然损失”。
它的名称来源于统计学中的最大似然估计（Maximum Likelihood Estimation, MLE），在深度学习中常用于分类任务。
'''

#!开始训练
for epoch in range(EPOCH):
  log.info('开始第 %d 轮训练', epoch + 1)
  my_fcnn_module.train(True)  # 设置模型为训练模式
  for i, (images, labels) in enumerate(dataloader):