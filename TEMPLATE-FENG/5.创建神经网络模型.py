# 使用pytorch框架， 自定义类，然后继承nn.Model , 完成构造函数和forward函数
# 创建一个用于数字识别的模型：
# 1， 使用全连接神经网络模型 FNN_Model.py （输入数据比较小的时候，可以选择只用全连接神经网络）
# 2. 使用卷积神经网络模型 CNN_Model.py（输入数据比较大的时候，
#     需要先使用卷积神经网络，然后展平， 在使用全连接神经网络）

import torch
class MyModel(torch.nn.Module):
    # 构造方法
    def __init__(self):
        super(MyModel , self ).__init__()
        print(f"执行自定义的神经网络的构造方法，该方法中定义神经网络中的每一层")

    # 前向传播的方法 , 给神经网络模型传入参数之后， 会自动调用前向传播的方法。
    def forward(self , x):
        print(f"执行前向传播的方法，输入是x。")
        return  x ** 2

if __name__ == "__main__":
    # 创建神经网络模型 ，创建模型对象的时候， 会自动调用构造函数
    module = MyModel()
    y = module(3) # 给模型传入参数， 则会调用前向传播函数
    print(y)

