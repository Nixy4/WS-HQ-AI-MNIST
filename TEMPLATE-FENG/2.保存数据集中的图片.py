#1.加载数据集，然后读取数据集中的内容，存储为图片
# 迭代器的使用
import torchvision
import matplotlib.pyplot as plt

#2.加载数据
train_data = torchvision.datasets.MNIST("",train=True,download=False)

#3.迭代器
inter = iter(train_data)
#4.保存数据
for i in range(16):
    image ,label = next(inter)
    print(image , label)
    # 设置保存路径
    image.save(f"imgs//{i}_{label}.png")
    plt.subplot(2,8,i+1) # 2行8列
    plt.title(label)
    plt.imshow(image)
plt.show()
