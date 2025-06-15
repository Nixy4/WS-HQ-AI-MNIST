# 张量的使用， 图片读取到程序中后，如何转换为方便使用的张量
from torchvision.transforms import ToTensor ,Resize ,Compose
from PIL import Image
import cv2 as cv

# 1. 加载图片，得到一个文件类型
image = Image.open("./imgs/0_5.png")
# 2. 张量转换器
to_tensor = ToTensor()
# 3.文件类型转换为张量
image_tensor = to_tensor(image)
# 4. 测试类型
print(type(image_tensor)) # <class 'torch.Tensor'>
print(type(image)) # <class 'PIL.PngImagePlugin.PngImageFile'>
# 5. 张量的使用
print(image_tensor.shape)
print(image_tensor)
# 6. 改变张量的size
m = Resize((14,14))
image_tensor = m(image_tensor)
print(image_tensor.shape)
