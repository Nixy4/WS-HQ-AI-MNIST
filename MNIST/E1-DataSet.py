import matplotlib.pyplot as plt
import torchvision
from logger import *

log.setLevel(logging.DEBUG)

log.info("导入(没有的话会下载)数据集")
train_data = torchvision.datasets.MNIST("./",train=True , download=True)
test_data = torchvision.datasets.MNIST("./",train=False , download=True)
'''
方法: torchvision.datasets.MNIST
参数1: 数据集存储路径, './'表示当前目录
参数2: train=True表示下载训练集, train=False表示下载测试集
参数3: download=True表示如果数据集不存在则下载
'''

# Display the first 5 images from the training dataset
log.info("显示前5张训练集图片")
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_data.data[i], cmap='gray')
    plt.title(f"Label: {train_data.targets[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()