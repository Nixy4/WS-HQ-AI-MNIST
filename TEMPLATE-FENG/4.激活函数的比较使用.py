# 不同的激活函数对预测值的影响
# 激活函数：
# 1.引入非线性，神经网络旨在处理各种复杂的非线性问题，而非线性模型的表达能力有限。
# 激活函数能为神经网络引入非线性因素，使网络可以学习和模拟各种复杂的非线性关系，通过激活函数的
# 使用，神经网络能拟合复杂的函数曲线，处理图像识别，语音识别等领域中的非线性问题。
# 2. 增加模型的表达能力 ， 激活函数使神经网络能够逼近任何复杂的函数，增加了模型的表示能力和灵活性
# 提高模型对复杂任务的处理能力
# 3. 决定神经元的输出状态，激活函数根据神经元的输入来确定其输出。
# 4. 缓解梯度消失和爆炸问题, 在神经网络训练中，梯度消失或爆炸（模型训练中的一种异常情况，反向
# 传播过程中，梯度值变得过大，呈指数级增长，导致模型无法正常训练）会导致训练困难。
# 5. 实现特征的稀疏性 ，像ReLU这样的激活函数，会使大量神经元的输出为0，从而使模型具有稀疏性，这有助于
# 减少模型的参数数量，降低模型的复杂度，提高模型的泛化能力，还能加快模型的训练速度，减少计算量。
import matplotlib.pyplot as plt
import torch
import numpy as np
# 1. 产生100个数据, 数据区间是-10~~10
x = np.linspace(-10,10,100)
x = torch.tensor(x) # 转换为张量
y = 0.3 * x  + 4;
fig ,(ax1,ax2,ax3)=plt.subplots(1,3)
ax1.plot(x,y,'r--')
relu = torch.nn.Sigmoid()
y_ = relu(y)
sigmoid = torch.nn.Tanh()
y__ = sigmoid(y)
ax2.plot(x,y_,'b--')
ax3.plot(x,y__,'g--')



'''
# 2. 使用RuLU激活函数
rulu = torch.nn.ReLU()
y = rulu(x)
plt.subplot(1,3,1)
plt.title("ReLU")
plt.plot(x,y,"b--")

# 3. 使用Sigmoid激活函数
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
plt.subplot(1,3,2)
plt.title("Sigmoid")
plt.plot(x,y,"r--")

# 4.调用Tanh激活函数
tanh = torch.nn.Tanh()
y = tanh(x)
plt.subplot(1,3,3)
plt.title("Tanh")
plt.plot(x,y,"g--")
'''
plt.show()
