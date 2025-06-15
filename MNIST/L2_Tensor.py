from torchvision.transforms import ToTensor ,Resize ,Compose
from PIL import Image
import os
from logger import *

img_path = './imgs/0_5.png'
img = Image.open(img_path)

to_tensor = ToTensor()
img_tensor = to_tensor(img)
print_tensor(img_tensor)
