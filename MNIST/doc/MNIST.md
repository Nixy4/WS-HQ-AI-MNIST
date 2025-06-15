# 0-词汇解释和相关概念

## MNIST

**MNIST** 是人工智能（尤其是计算机视觉和机器学习）中最经典的**手写数字识别数据集**，其名称由以下单词缩写组成：

- **M**odified（改进的）
- **N**ational（美国的）
- **I**nstitute（研究院）
- **S**tandards（标准）
- **T**echnology（与技术）

全称是 **Modified National Institute of Standards and Technology database**（改进版美国国家标准与技术研究院数据库）。

## Tensor

张量

## ToTensor

张量转换器

## Neural Network Activation Functions

神经网络激活函数

```python
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
x = np.linspace(-5, 5, 500)

# 计算各激活函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_swish = swish(x)
y_gelu = gelu(x)

# 创建图形
plt.figure(figsize=(14, 10))

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
plt.xlim([-5, 5])
plt.ylim([-2, 2])

# 显示图形
plt.tight_layout()
plt.show()

```





# 1-开发环境

- IDE

  PyCharm

- Python版本

  Python 3.12

- Python包

  - torch 2.7.1
  - torchvision 0.22.1
  - matplotlib 3.10.3



# 2-MNIST数据集准备


