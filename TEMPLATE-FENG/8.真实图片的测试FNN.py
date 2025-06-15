#  读取imgs中的一张图片，然后预测图片是数字多少
from PIL import Image,ImageOps
from torchvision.transforms import ToTensor ,Resize ,Compose
import torch
from FNN_Model import FNN_Model

# 1. 打开一张图片
image = Image.open("imgs/1_0.png")
#image = image.convert("L") # 转换为灰度图
#image = ImageOps.invert(image) # 转为黑底白字
to_tensor = Compose({
    ToTensor(),
    Resize((28,28))
})
#通道 高 宽
image_tensor = to_tensor(image)
print(image_tensor.shape)
# 2.加载模型， 预测
model = FNN_Model()
model.load_state_dict(torch.load("FNN_Model.pt"))
output = model(image_tensor)
# 3.得到预测结果
print(f"{torch.argmax(output)}")
