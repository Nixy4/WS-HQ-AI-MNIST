import tkinter as tk
from tkinter import Canvas , Label ,Button
from torchvision.transforms import ToTensor
from PIL import Image , ImageDraw , ImageOps
import torch
from CNN_Model import CNN_Model

class HandwritingApp:
    #构造函数
    def __init__(self,root):
        self.root = root
        self.root.title = "MNIST数字识别"
        # 创建Canvas
        self.canvas = Canvas(root,width=256,height=256,bg='white')
        # 使用pack布局管理器放置canvas
        # padx 水平方向的边距（组件左右两侧的空白空间）， pady垂直方向的边距
        self.canvas.pack(padx=10,pady=10)
        #绑定鼠标事件
        self.canvas.bind("<Button-1>",self.on_draw_start)
        self.canvas.bind("<B1-Motion>",self.on_draw_move)

        #创建一个空的PIL图像用于保存绘制结果
        self.drawing = Image.new("RGB",(256,256),
                                 'white')
        self.draw = ImageDraw.Draw(self.drawing)
        self.img_tensor = None

        # 识别按钮
        self.save_button = Button(root,text="识别",
                                     command=self.on_save_button_clicked)
        self.save_button.pack(pady=20)

        # 清楚发按钮
        self.clear_button = Button(root , text="清除"
                                      ,command=self.on_clear_button_clicked)
        self.clear_button.pack(pady=10)
        # 显示预测结果的标签
        self.prediction_label = Label(root,text="" , width=20)
        self.prediction_label.pack(pady=20)

        # 加载模型参数
        self.model = CNN_Model()
        state_dict = torch.load("CNN_Model.pt")
        self.model.load_state_dict(state_dict)


    def on_draw_start(self,event):
        self.lastx , self.lasty = event.x , event.y
    def on_draw_move(self,event):
        x,y = event.x ,event.y
        # 自己单独创建的一张特征图片上的内容修改
        self.draw.line((self.lastx , self.lasty, x,y),
                       fill="black",width=10)
        # 这个是在画布上绘制的图片，让用户可以看见的内容
        self.canvas.create_line(self.lastx , self.lasty , x, y,
                                fill="black" , width=10 , capstyle=tk.ROUND
                                ,smooth=tk.TRUE , splinesteps=36)
        # 点的位置在不断变化
        self.lastx , self.lasty = x, y


    def save_as_tensor(self):
        img_gray = self.drawing.convert("L")
        img_gray = ImageOps.invert(img_gray)
        img_gray = img_gray.resize((28,28))
        to_tensor = ToTensor()
        img_tensor = to_tensor(img_gray).float()
        self.img_tensor = img_tensor.unsqueeze(0) # 升维
    def on_save_button_clicked(self):
        '''
        function:这个函数识别按钮对应的函数
        '''
        self.save_as_tensor()
        outputs = self.model.forward(self.img_tensor)
        prediction = torch.argmax(outputs)
        self.prediction_label.config(text=f"识别结果：{int(prediction)}")


    def on_clear_button_clicked(self):
        self.canvas.delete("all")
        self.drawing = Image.new("RGB",(256,256),'white')
        self.draw = ImageDraw.Draw(self.drawing)
        self.img_tensor = None
        self.prediction_label.config(text="")
root = tk.Tk()
app = HandwritingApp(root)
root.mainloop()