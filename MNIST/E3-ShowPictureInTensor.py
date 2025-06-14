import torch
from torchvision.transforms import v2 # 导入v2版本的转换器API, 本小结使用 Compose 和 Resize
from PIL import Image # Python Imaging Library, Python图像库, 用于处理图像文件
from PIL import ImageFile
import os
from logger import *

'''读取'./imgs'文件夹中的所有图片文件'''
img_dir = './imgs'
img_fileList = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
log.debug(f'imgs中的文件:{img_fileList}')

'''打开img_fileList中第一个元素的文件'''
img_path = os.path.join(img_dir, img_fileList[0])
img = Image.open(img_path)
log.info(f'打开图片文件: {img_path}')

'''1.定义 图像->张量 的转换器'''
img_transform = v2.Compose([
  v2.ToImage(), # 保留Alpha通道
  v2.ToDtype(torch.float32, scale=True) # 将像素值转换为浮点数并缩放到[0, 1]范围
])

'''2.将图片转换为张量'''
img_tensor = img_transform(img)

'''4.打印类型'''
log.debug(f'img类型: {type(img)}')
log.debug(f'img_tensor类型: {type(img_tensor)}')

'''5.打印实际数据'''
#显示原始图片
# log.info("显示原始图片:")
# img.show()
#打印原始图像数据
def log_image(image:ImageFile.ImageFile):
  # 将PIL图片转为numpy数组
  import numpy as np
  arr = np.array(image)
  # 如果是彩色图像，取第一个通道（灰度或R通道）
  if arr.ndim == 3:
    arr = arr[..., 0]
  lines = []
  for row in arr:
    line = ' '.join(f'{int(v):03d}' for v in row)
    lines.append(line)
  result = '\n'.join(lines)
  result = '\n'+ result
  log.debug(result)
log.info("打印原始图像数据:")
log_image(img)
#打印张量数据
def log_tensor(tensor: torch.Tensor):
  # 假设tensor为[1,28,28]或[28,28]，先去掉batch和通道维
  arr = tensor.squeeze().tolist()
  lines = []
  for row in arr:
    # 格式化为字符串，保留2位小数
    line = ' '.join(f'{v:6.3f}' for v in row)
    lines.append(line)
  result = '\n'.join(lines)
  result = '\n' + result
  log.debug(result)
log.info("打印张量数据:")
log_tensor(img_tensor)

'''6.打印张量形状'''
log.info(f'img_tensor形状: {img_tensor.shape}')

'''7.改变张量形状'''
resize_transform = v2.Resize((28, 28))  # 将图像大小调整为28x28
img_tensor = resize_transform(img_tensor)
log.info(f'调整后的img_tensor形状: {img_tensor.shape}')