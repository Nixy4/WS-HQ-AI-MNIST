import logging

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import PIL
import PIL.Image as Image
import numpy as np

class CNN0(nn.Module):

  def __init__(self):
    super(CNN0, self).__init__()
    self.Sequential = nn.Sequential(
      nn.Conv2d(1, 10, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(10, 20, 5),
      nn.Flatten(),
      nn.Linear(1620, 100),
      nn.ReLU(),
      nn.Linear(100, 10),
      nn.Softmax(dim=1)
    )
    '''
    0.    输入Tensor: 28x28个像素,每个像素有1个通道(灰度图像)
          shape = (1, 28, 28)
    
    1.    Conv2d: 第一次卷积
            in_channels=1: 输入图像的通道数为1(灰度图像)
            out_channels=10:  卷积核数量为10, 输出特征图的通道数为10, 每个卷积核提取不同特征, 具体特征由卷积核学习得到
                              对输入的每个通道，使用多个可学习的卷积核（权重矩阵）在空间上滑动（卷积操作）
                              每个卷积核与输入的局部区域做逐元素乘法并求和，得到一个输出值，形成特征图
                              所有卷积核分别提取不同的特征，输出多个特征图（即输出通道）
                              卷积核参数通过训练自动优化，以提取有用特征
            kernel_size=3: 卷积核大小为3x3
            shape: (1, 28, 28) -> (10, 26, 26)
            
    2.    ReLU: 对卷积输出进行非线性变换
            shape: (10, 26, 26) -> (10, 26, 26)
    
    3.    MaxPool2d: 减少特征图的空间尺寸, 提取主要特征, 降低计算复杂度
            kernel_size=2: 池化窗口大小为2x2
            stride=2: 池化步幅为2, 每次移动2个像素
            shape: (10, 26, 26) -> (10, 13, 13)
            
    4.    Conv2d: 第二次卷积
            in_channels=10: 输入特征图的通道数为10(来自上一步的输出)
            out_channels=20: 卷积核数量为20, 输出特征图的通道数为20
            kernel_size=5: 卷积核大小为5x5
            shape: (10, 13, 13) -> (20, 9, 9)
            
    5.    Flatten: 将多维特征图展平为一维向量, 准备输入到全连接层
            shape: (20, 9, 9) -> (20 * 9 * 9) = (1620)
            
    6.    Linear: 第一次全连接, 将展平后的特征向量映射到输出空间
            in_features=1620: 输入特征数为1620(来自Flatten层的输出)
            out_features=100: 输出特征数为100, 这可以是分类任务的类别数或其他任务的输出维度
            shape: (1620) -> (10)
    7.    ReLU: 对全连接层输出进行非线性变换
            shape: (100) -> (100)
    8.    Linear: 第二次全连接, 将100个特征映射到10个输出类别(0~9)
            in_features=100: 输入特征数为100(来自上一步的输出)
            out_features=10: 输出特征数为10, 通常对应于10个类别的分类任务
            shape: (100) -> (10)
    9.    Softmax: 将输出转换为概率分布, 所有输出值的和为1
            dim=1: 在第1维(类别维度)上进行Softmax操作
            shape: (10) -> (10)
    '''

  def forward(self, x):
    return self.Sequential(x)

img_path = './imgs/0_5.png'
img_obj = Image.open(img_path)
logging.info('图片{}已打开'.format(img_path))

def log_img(img: PIL.Image.Image):
    arr = np.array(img)
    if arr.ndim == 2:
        # 灰度图像
        print("[灰度] 通道:")
        for row in arr:
            print(' '.join(f'{int(v):03d}' for v in row))
    elif arr.ndim == 3:
        channels = arr.shape[2]
        for c in range(channels):
            print(f"[通道 {c}] 数据:")
            for row in arr[..., c]:
                print(' '.join(f'{int(v):03d}' for v in row))
    else:
        print("未知图像格式")
log_img(img_obj)