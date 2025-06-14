import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 定义各种激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# 生成x值范围
x = np.linspace(-10, 10, 1000000)

# 计算各激活函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_swish = swish(x)
y_gelu = gelu(x)

# 创建图形
plt.figure(figsize=(20, 20))

# 绘制各激活函数曲线
plt.plot(x, y_sigmoid, label='Sigmoid', linewidth=2)
plt.plot(x, y_tanh, label='Tanh', linewidth=2)
plt.plot(x, y_relu, label='ReLU', linewidth=2)
plt.plot(x, y_leaky_relu, label='Leaky ReLU (α=0.01)', linewidth=2)
plt.plot(x, y_elu, label='ELU (α=1.0)', linewidth=2)
plt.plot(x, y_swish, label='Swish (β=1.0)', linewidth=2)
plt.plot(x, y_gelu, label='GELU', linewidth=2)

# 添加图形元素
plt.title('Neural Network Activation Functions', fontsize=16)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# 设置坐标轴范围
# plt.xlim([-5, 5])
# plt.ylim([-2, 2])

# 显示图形
plt.tight_layout()
plt.show()
