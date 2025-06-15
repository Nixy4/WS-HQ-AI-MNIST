import torch
import torchvision
from CNN_Model import CNN_Model
#1.定义超参数
EPOCH = 3
BATCH_SIZE=10
LEARNING_BATE=0.0001 # 0.001的时候，准确率很低， 0.0001的时候准确率提高了很多。
#2.加载数据集
to_tensor = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST("",transform=to_tensor,train=True,download=False)
test_set = torchvision.datasets.MNIST("",transform=to_tensor,train=False,download=False)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
#3.定义模型，优化器，损失函数
model = CNN_Model()
loss_fun = torch.nn.NLLLoss()
opt = torch.optim.Adam(model.parameters(),lr=LEARNING_BATE)
#4.评估函数
def eval(model,test_loader):
    #1.开启评估
    model.eval()
    #2.计算正确率
    if torch.no_grad():
        total = 0
        correct = 0
        for images , labels in test_loader:
            outputs = model(images)
            for i , output in enumerate(outputs):
                if torch.argmax(output) == labels[i]:
                    correct += 1
                total += 1
    return  correct / total
#5.训练模型
for i in range(EPOCH):
    print(f"第{i+1}次训练纪元：")
    #1.开启训练
    model.train(True)
    #2.循环训练
    for i,(images , labels) in enumerate(train_loader):
        #梯度清零
        '''
        遇到的错误： 代码写成了torch.no_grad() ， 这里应该是对优化器的梯度清零
        由于没有对优化器梯度清零，导致训练的模型，准去率较低，并且需要把学习率
        设置得很小，否则模型的准去率很低。   
        '''
        opt.zero_grad()
        #前向传播
        outputs = model(images)
        #计算损失
        loss = loss_fun(outputs , labels)
        #反向传播
        loss.backward()
        #优化参数
        opt.step()
    #3.测试模型
    accurate = eval(model , test_loader)
    print(f"{i+1}次训练的正确率是：{accurate}")
#6.保存模型
# ***这里加个判断，读取之前的训练结果，保存准确率最高的模型
torch.save(model.state_dict(), "CNN_Model.pt")