import torch
import torch.nn as nn
from torchsummary import summary
class FNN_Model(nn.Module):
    #构造函数
    def __init__(self):
        super(FNN_Model , self).__init__()
        self.sequential = nn.Sequential(
            # a.张量展平： MNIST中的图片原结构为（1，28，28），展平后变为一个
            # 一维数组，张量为(1,28*28)
            nn.Flatten(),
            # b.经过全连接层
            nn.Linear(1*28*28,100),
            # c.激活函数
            nn.ReLU(),
            # d.经过全连接层
            nn.Linear(100,10),
            # # e.归一化处理: 使多分类问题的概率值和为1.
            # nn.LogSoftmax(dim=1) # LogSoftmax函数，将上一步的输出特征，映射到（0，1）范围内
        )

    def forward(self , x):
        return self.sequential(x);
if __name__ == '__main__':
    model = FNN_Model()
    summary(model,(1,28,28))
