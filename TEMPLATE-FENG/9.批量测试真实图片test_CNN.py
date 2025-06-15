# 批量读取准备好的真实图片，然后对这些图片进行转换，然后处理为一个数据集，最后批量预测，遍历预测结果
import torch
from PIL import Image , ImageOps
from torchvision.transforms import ToTensor , Resize , Compose
from CNN_Model import CNN_Model

#1.图片路径list

img_list = ["MNIST/test/0.png","MNIST/test/1.png","MNIST/test/2.png",
            "MNIST/test/3.png","MNIST/test/4.png","MNIST/test/5.png",
            "MNIST/test/6.png","MNIST/test/7.png"]
# 2. 转换器
to_tengsor = Compose([ToTensor()])
# 3.图片转换成的张量集合(list类型)
img_tensor_list=[]
for img in img_list:
    image = Image.open(img)
    image_tensor = to_tengsor(image)
    img_tensor_list.append(image_tensor)
print(img_tensor_list[0].shape)

#4.图片张量集合转换为一个张量对象
img_tensor_obj = torch.stack(img_tensor_list)
#5. 加载模型，预测
model = CNN_Model()
model.load_state_dict(torch.load("CNN_Model.pt"))
outputs = model(img_tensor_obj)
# 6. 输出预测结果
for i,output in enumerate(outputs):
    print(f"{i} ---- {torch.argmax(output)}")


