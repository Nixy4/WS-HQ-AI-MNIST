# 批量读取准备好的真实图片，然后对这些图片进行转换，然后处理为一个数据集，最后批量预测，遍历预测结果
import torch
from PIL import Image , ImageOps
from torchvision.transforms import ToTensor , Resize , Compose
from CNN_Model import CNN_Model

#1.图片路径list
'''
这个问题找了很久的错误， 因为使用的{}是set类型， 导致遍历的时候获取到
数据值是无序的，导致一致识别率很低。 后面参考别人使用的[],是list类型
，遍历是有序的， 这样才可以用序号对应着图片上的数据值，来对比识别是否准确。
img_list = {"MNIST/trueImgs/0.png",
            "MNIST/trueImgs/1.png",
            "MNIST/trueImgs/2.png",
            "MNIST/trueImgs/3.png",
            "MNIST/trueImgs/4.png",
            "MNIST/trueImgs/5.png",
            "MNIST/trueImgs/6.png",
            "MNIST/trueImgs/7.png",
            "MNIST/trueImgs/8.png",
            "MNIST/trueImgs/9.png"}
'''
img_path_list = [
    "MNIST/trueImgs/0.png",
    "MNIST/trueImgs/1.png",
    "MNIST/trueImgs/2.png",
    "MNIST/trueImgs/3.png",
    "MNIST/trueImgs/4.png",
    "MNIST/trueImgs/5.png",
    "MNIST/trueImgs/6.png",
    "MNIST/trueImgs/7.png",
    "MNIST/trueImgs/8.png",
    "MNIST/trueImgs/9.png",
]

# 2. 转换器
to_tengsor = Compose([Resize((28,28)),ToTensor()])
# 3.图片转换成的张量集合(list类型)
img_tensor_list=[]
for img in img_path_list:
    print(img ,"=================")
    # 转变为灰度图， 转换背景色， 转换为张量
    image = Image.open(img)
    image = image.convert("L")
    image = ImageOps.invert(image)
    image_tensor = to_tengsor(image)
    img_tensor_list.append(image_tensor)

#4.图片张量集合转换为一个张量对象
img_tensor_obj = torch.stack(img_tensor_list)
print(img_tensor_obj.shape)

#5. 加载模型，预测
model = CNN_Model()
model.load_state_dict(torch.load("CNN_Model.pt"))
outputs = model(img_tensor_obj)

prediction = torch.argmax(outputs,dim=1)
print(prediction)

# 6. 输出预测结果
for i ,output in enumerate(outputs):
    print(i , torch.argmax(output))


