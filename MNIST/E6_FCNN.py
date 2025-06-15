from logger import log
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

class MyFCNN(nn.Module):

  #! 定义构造函数
  def __init__(self):
    # 调用父类的构造函数
    super(MyFCNN, self).__init__()
    #初始化顺序容器, 向容器中添加层, 容器会自动将层按添加顺序连接起来
    self.sequential = nn.Sequential(
      nn.Flatten(),#1.张量展平层
      nn.Linear(784,100), #2.全连接层
      nn.ReLU(), #3.激活函数
      nn.Linear(100, 10), #4.全连接层
      nn.LogSoftmax(dim=1) #5.对数Softmax层
    )
    log.info("MyFCNN初始化成功")

    '''
    1.  nn.Flatten()：输入图片分辨率为28x28, 像素数量为28x28=784, 该层将输入张量展平为一维向量。
        原本输入张量为(1, 28, 28)，展平后变为(1, 784)。
    2.  nn.Linear(784, 100)：全连接层，将展平后的784个输入特征转换为(总结为)100个输出特征。
    3.  nn.ReLU()：激活函数层，使用ReLU激活函数对100个输出特征进行非线性变换。
    4.  nn.Linear(100, 10)：另一个全连接层，将上一层输出的100个特征转换为10个输出特征，通常对应于10个类别的分类任务。
    5.  nn.LogSoftmax(dim=1)：对数Softmax层，将10个输出特征转换为10个输出值, 10个输出值和为1，适用于多分类任务。
    '''

  #! 定义前向传播函数
  def forward(self, x):
    return self.sequential(x)

# 实例化网络 张量转换器
my_fcnn_module = MyFCNN()

# 定义图像转换器
img_transform = v2.Compose([
  v2.ToImage(), # 保留Alpha通道
  v2.ToDtype(torch.float32, scale=True) # 将像素值转换为浮点数并缩放到[0, 1]范围
])

img_path = './imgs/0_5.png'
img_file = Image.open(img_path)
img_tensor = img_transform(img_file)

#输入数据
y = my_fcnn_module.forward(img_tensor)
# 打印输出
log.info(f'输出结果:\n{y}')


