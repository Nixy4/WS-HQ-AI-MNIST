import torch.optim
import torchvision
from FNN_Model import FNN_Model
from torch.utils.data import DataLoader
import numpy as np

# 训练模型
# 1.定义超参数
EPOCH = 3
BATCH_SIZE = 10
LEARNING_RATE = 0.001
# 2.加载数据集: 需要做张量的转换
to_tensor = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST("",transform=to_tensor, train=True,download=False)
test_data = torchvision.datasets.MNIST("",transform=to_tensor,train=False,download=False)
train_loader = DataLoader(train_data , batch_size= BATCH_SIZE , shuffle=True)
test_loader = DataLoader(test_data , batch_size= BATCH_SIZE , shuffle=True)
# 3.创建自定义的神经网络模型，优化器，损失函数
model = FNN_Model()
opt = torch.optim.Adam(model.parameters() , lr = LEARNING_RATE)
loss_fun = torch.nn.NLLLoss()
# 4.评估函数
def eval(model,test_loader):
    # 1.开启评估
    model.eval()
    # 2.评估时候不要调整训练参数（不计算梯度）
    with torch.no_grad():
        # 测试样本总数，正确数
        total = 0
        correct = 0
        # 遍历测试集
        for images ,labels in test_loader:
            # 预测数据
            outputs = model(images)
            # 检查预测结果
            for i ,output in enumerate(outputs):
                # print(f"np.argmax(output) : {np.argmax(output)}")
                # print(f"torch.argmax(output):{torch.argmax(output)}")
                if np.argmax(output) == labels[i]:
                    correct += 1
                total += 1
    return correct / total
# 5.训练模型
for i  in range(EPOCH):
    print(f"训练纪元：{i+1}次")
    #开启训练
    model.train(True)
    #遍历训练集
    for i ,(images , labels) in enumerate(train_loader):
        #梯度清零（优化器的操作）
        opt.zero_grad()
        #前向传播
        outputs = model.forward(images)
        # 计算损失
        loss  = loss_fun(outputs , labels)
        # 反向传播
        loss.backward()
        # 优化参数
        opt.step()
    # 测试模型
    accurate =  eval(model,test_loader)
    print(f"{i+1}/{EPOCH}次测试结果的正确率是：{accurate}")
# 6.保存模型
torch.save(model.state_dict(),"FNN_Model.pt")