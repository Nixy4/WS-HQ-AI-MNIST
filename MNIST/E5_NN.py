import torch
from logger import log

#!定义一个简单的神经网络类
class MyModule(torch.nn.Module):
    #构造函数
    def __init__(self):
        super(MyModule, self).__init__() #调用父类的构造函数
        log.info('构造函数')
    @staticmethod
    def forward(x: int):
        log.info('前向传播')
        return x**2
#!实例化网络
my_module = MyModule()
#!输入数据
y = my_module.forward(2)
#!打印输出
log.info(f'输出结果: {y}')

#!定义一个<用于分辨颜色是亮色还是暗色的>神经网络模块
class ColorLightness(torch.nn.Module): #输入会读者判断是亮色还是暗色
    def __init__(self):
        super(ColorLightness, self).__init__()
    @staticmethod
    def forward(rgb: tuple[3]):
        _y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return '亮色' if _y > 127 else '暗色'
color_lightness_module = ColorLightness()
# 输入RGB颜色值
red = (255, 255, 0)  # 绿色
black = (0, 0, 0) # 黑色
result = color_lightness_module.forward(red)
log.info(f'颜色 {red} 是 {result}')
result = color_lightness_module.forward(black)
log.info(f'颜色 {black} 是 {result}')

#!定义一个瓦学弟神经网络模块
class WaXueDi(torch.nn.Module):
    def __init__(self):
        super(WaXueDi, self).__init__()
        log.info('构造函数')
    @staticmethod
    def forward(teammate_type: str)->str:
        if teammate_type in ['地雷妹', '御姐', '少萝','小仙女']:
            return '妈妈'
        else:
            return '哥们'
waxuedi_module = WaXueDi()
# 输入瓦学弟的队友
teammateType= '地雷妹'  # 可以修改为其他类型
result = waxuedi_module.forward(teammateType)
log.info(f'瓦学弟称呼他的{teammateType} 队友为 {result}')