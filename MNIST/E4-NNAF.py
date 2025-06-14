import matplotlib.pyplot as plt
import torch
import numpy as np

'''准备输入值np数组'''
x_values = np.linspace(-10, 10, 100)
x_values = torch.tensor(x_values)

plt.figure(figsize=(20, 20))

'''1.Sigmoid激活函数: φ(z) = 1 / (1 + exp(-z))'''
nnaf_sigmoid = torch.nn.Sigmoid()
y_values = nnaf_sigmoid(x_values)
#绘制
plt.subplot(3, 3, 1)
plt.title("Sigmoid")
plt.plot(x_values, y_values, "r")

'''2.Tanh激活函数: φ(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))'''
nnaf_tanh = torch.nn.Tanh()
y_values = nnaf_tanh(x_values)
#绘制
plt.subplot(3, 3, 2)
plt.title("Tanh")
plt.plot(x_values, y_values, "g")

'''3.Softmax激活函数: φ(z_i) = exp(z_i) / sum(exp(z_j)) for j in range(n)'''
nnaf_softmax = torch.nn.Softmax(dim=0)  # dim=0 for vector input
y_values = nnaf_softmax(x_values)
#绘制
plt.subplot(3, 3, 3)
plt.title("Softmax")
plt.plot(x_values, y_values, "c")

'''4.ReLU激活函数: φ(z) = max(0,z)'''
nnaf_relu = torch.nn.ReLU()
y_values = nnaf_relu(x_values)
#绘制
plt.subplot(3, 3, 4)
plt.title("ReLU")
plt.plot(x_values, y_values, "b")

'''5.Leaky ReLU激活函数: φ(z) = z if z > 0 else α * z'''
nnaf_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
y_values = nnaf_leaky_relu(x_values)
#绘制
plt.subplot(3, 3, 5)
plt.title("Leaky ReLU")
plt.plot(x_values, y_values, "k")

'''6.PReLU激活函数: φ(z) = z if z > 0 else α * z, where α is a learnable parameter'''
nnaf_prelu = torch.nn.PReLU()
y_values = nnaf_prelu(x_values.float()).int()  # 保证类型一致
#绘制
plt.subplot(3, 3, 6)
plt.title("PReLU")
plt.plot(x_values, y_values, "o")

'''7.ELU激活函数: φ(z) = z if z > 0 else α * (exp(z) - 1)'''
nnaf_elu = torch.nn.ELU()
y_values = nnaf_elu(x_values)
#绘制
plt.subplot(3, 3, 7)
plt.title("ELU")
plt.plot(x_values, y_values, "y")

'''8.GELU激活函数: φ(z) = z * P(z) where P(z) is the cumulative distribution function of the standard normal distribution'''
nnaf_gelu = torch.nn.GELU()
y_values = nnaf_gelu(x_values)
#绘制
plt.subplot(3, 3, 8)
plt.title("GELU")
plt.plot(x_values, y_values, "c")
'''9.Swish激活函数: φ(z) = z * sigmoid(z)'''
nnaf_swish = torch.nn.SiLU()  # Swish is implemented as SiLU in PyTorch
y_values = nnaf_swish(x_values)
#绘制
plt.subplot(3, 3, 9)
plt.title("Swish")
plt.plot(x_values, y_values, "m")

plt.show()