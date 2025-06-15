# 卷积神经网络
import torch.nn as nn
from torchsummary  import  summary
class CNN_Model(nn.Module):
    # 构造函数
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 通过神经网络序列来定义神经网络中的每一层
        self.sequential = nn.Sequential(
            # 第一次卷积，将灰度图按3*3的卷积核进行卷积，输出10个特征。
            # 28*28通过3*3的卷积核，大小变为(28-3+1),即26*26
            nn.Conv2d(1, 10, 3),  # nn.Conv2d(输入通道数，输出通道数，卷积核大小)二维卷积
            # 激活函数
            nn.ReLU(),
            # 最大值池化。将26*26的图片池化，获取每2*2窗口中的最大值，每次跳过2个长度，最终长和宽都小一半，
            # 得到13*13的特征图
            nn.MaxPool2d(2, 2),  # nn.MaxPool2d(池化窗口大小，步长)
            # 第二次卷积，将13*13的特征图，按5*5的卷积核卷积，输出20个特征
            # 13*13通过5*5的卷积核，大小变为(13-5+1),即9*9
            nn.Conv2d(10, 20, 5),
            # 激活函数
            nn.ReLU(),
            # 展平
            nn.Flatten(),
            # 经过全连接层，将9*9*20(上一步卷积后的特征数)， 输出100个新特征
            nn.Linear(9 * 9 * 20, 100),
            # 激活函数
            nn.ReLU(),
            # 全连接层，得到10个最终的输出，即0-9的概率
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)  # 归一化
        )
        '''
        self.sequential = nn.Sequential(
            #a卷积，激活函数，池化
            # 输入通道1，输出10，卷积核大小为3*3
            # 卷积运算后的大小为： 28-3+1 ，即 26*26
            nn.Conv2d(1,10,3),
            nn.ReLU(),
            # 池化后大小为13*13
            nn.MaxPool2d(2,2),
            #b.卷积，激活函数，展平
            # 卷积运算后大小为： 13-5+1，即9*9
            nn.Conv2d(10,20,5),
            nn.ReLU(),
            # 展平后的大小为 9*9*20
            nn.Flatten(),
            #c.全连接,激活函数
            nn.Linear(9*9*20,100),
            nn.ReLU(),
            #d.全连接，激活函数
            nn.Linear(100,10), # 输出结果后，不能添加激活函数， 否则结果会被再次改变。
            # 这里的激活函数导致模型的输出结果出现偏差，输出结果之后不能出现激活函数了。
            nn.ReLU(), # 
            #e.归一化
            nn.LogSoftmax(dim=1)

        )'''

    # 前向传播的函数
    def forward(self, x):
        return self.sequential(x)
if __name__ == '__main__':
    model = CNN_Model()
    summary(model,(1,28,28))