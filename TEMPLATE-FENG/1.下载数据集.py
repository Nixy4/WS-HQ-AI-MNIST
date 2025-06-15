# 1 需要使用到torchvision，matplotlib,如果没有就需要先安装
# pip install torchvision ,  pip install matplotlib
import matplotlib.pyplot as plt
import torchvision

# 2.下载数据集(训练集，测试集)
# train=True 训练集， train=False 测试集
# download=True 下载数据， download=False ,加载本地数据集
train_data = torchvision.datasets.MNIST("",train=True , download=False)
test_data = torchvision.datasets.MNIST("",train=False , download=False)

# 3.读取其中一个数据值，显示这个数据
image,label = train_data[0] # image --train_data[0][0] , label -- train_data[0][1]
print(image , label)
print(train_data[0][1])
plt.imshow(image)
plt.title(label)
plt.show()