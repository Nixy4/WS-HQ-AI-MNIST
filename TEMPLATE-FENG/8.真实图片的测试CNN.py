#  读取imgs中的一张图片，然后预测图片是数字多少
#  ** 卷积神经网络不能处理单个样本数据 **
from PIL import Image,ImageOps
from torchvision.transforms import ToTensor ,Resize ,Compose
import torch
from CNN_Model import CNN_Model
from FNN_Model import FNN_Model

# 1. 打开一张图片
image = Image.open("imgs/8_1.png")
#image = image.convert("L") # 转换为灰度图
#image = ImageOps.invert(image) # 转为黑底白字
to_tensor = Compose({
    ToTensor(),
    Resize((28,28))
})
# 批次 通道 高 宽 ， . 这里因为用到卷积神经网络，需要把单个样本数据转换为可以批量处理的形式
# unsqueeze(0) 是PyTorch中用于张量操作的函数， 作用是在指定位置插入一个维度为1的新维度，
# ‘0’代表在第0维度，即最左边维度插入。
image_tensor = to_tensor(image)
print(f"{image_tensor.shape}======") # [1, 28, 28]
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.shape,'======')# [1, 1, 28, 28]
# 2.加载模型， 预测
model = FNN_Model()
model.load_state_dict(torch.load("FNN_Model.pt"))
output = model(image_tensor)
print(f"{torch.sum(input=output)}")
# 3.得到预测结果
print(f"{torch.argmax(output)}")
