import matplotlib.pyplot as plt
import torchvision
from logger import *
import os

#检测imgs文件夹是否存在, 如果不存在则创建
if not os.path.exists("./imgs"):
    os.makedirs("./imgs")
    log.info("创建imgs文件夹")

train_data = torchvision.datasets.MNIST("./", train=True, download=True)
iterator = iter(train_data)

for i in range(10): # 循环10次, 导出10张图片
    # 获取迭代器中的下一个元素, 并将其解包为图像和标签
    image, label = next(iterator)
    # 图片添加到plot中
    plt.subplot(2, 5, i + 1) # 创建2行5列的子图, i+1表示当前图像的位置
    plt.imshow(image, cmap='gray') # 显示图像, cmap='gray'表示使用灰度色图
    plt.title(f"Label: {label}") # 设置标题为标签
    plt.axis('off') # 关闭坐标轴显示
    # 图片存储 ++++++++++++++++++++++++++
    filePatch = f"./imgs/{i}_{label}.png" # 定义文件路径, 包含索引和标签
    image.save(filePatch)
    log.debug("保存图像到文件: %s", filePatch)

