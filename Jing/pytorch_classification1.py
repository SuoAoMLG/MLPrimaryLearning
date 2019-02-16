import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import numpy
import csv
import pandas as pd
import torch.utils.data as Data
BATCH_SIZE = 99
LR = 0.05

#这个类用来导入数据
class DataInit:
    def __init__(self,DataSource,author):                   #初始函数用来初始换类里面的固有的属性和方法                                               #self参数似乎是伴随整个类的指向自己参数
        self.DataSource=DataSource
        self.author=author
                                                            #定义在类里面的方法（函数）"E:\\MachineLearning\\pytorch\\train.csv"
    def DataPrepare(self):
            self.passengerId = pd.read_csv(self.DataSource, usecols=['PassengerId'])
            self.name = pd.read_csv(self.DataSource, usecols=['Name'])
            self.survived = pd.read_csv(self.DataSource, usecols=['Survived'])
            self.input = pd.read_csv(self.DataSource, usecols=['Pclass','Sex','Age','SibSp','Parch','Fare'])
            
            
datatrain=DataInit("E:\\MachineLearning\\pytorch\\train.csv","Jing")
datatrain.DataPrepare()
y=datatrain.survived.as_matrix().reshape((-1,1))
y=torch.from_numpy(y).type(torch.LongTensor).squeeze()
x=datatrain.input.as_matrix()
x=torch.from_numpy(x).type(torch.FloatTensor)
#导入数据，注意对于y标记量要加squeeze()，要去掉维度数为1的值（其实就是去一个值，转化成神经网络可接受的量）
'''
print(y)
print(y.shape)
print(x)
'''
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

#这是定义神经网络的输入features，隐藏层数，输出数量的Net类
net = Net(6,15,2)           #输出层输出的是二维向量，但是要通过softmax激励函数转化为概率

optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.1, 0.11))
#optimizer = torch.optim.SGD(net.parameters(),lr=LR)      优化器选择，这里我们选择Adam
loss_func = torch.nn.CrossEntropyLoss() 
  
#批量处理数据，总共epoch 15次，每次训练9批数据，每一批数据batchsize 99 个
def show_batch():
    for epoch in range(15):   # train entire dataset 15 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            out = net(batch_x)
            loss = loss_func(out,batch_y)       #其标签必须为0~n-1，而且必须为1维的，如果设置标签为[nx1]的，则会出现【error】RuntimeError: multi-target not supported at错误
            optimizer.zero_grad()               #将最原始的梯度清零
            loss.backward()                     #直接计算梯度（封装好的函数），执行反向回归算法
            optimizer.step()                    #计算梯度，然后开始梯度下降
            prediction = torch.max(F.softmax(out), 1)[1] 
            pred_y = prediction.data.numpy().squeeze()
            target_y = batch_y.data.numpy()
            accuracy = sum(pred_y == target_y)/99.  # 预测中有多少和真实值一样
            print(epoch)
            print(step)
            print(accuracy)
            print(target_y)
            print(pred_y)

if __name__ == '__main__':
    show_batch()


datatest_input = pd.read_csv("E:\\MachineLearning\\pytorch\\test.csv", usecols=['Pclass','Sex','Age','SibSp','Parch','Fare'])
x1=datatest_input.as_matrix()
x1=torch.from_numpy(x1).type(torch.FloatTensor)
passergerIDtest=pd.read_csv("E:\\MachineLearning\\pytorch\\test.csv", usecols=['PassengerId'])
passergerIDtest1=passergerIDtest.as_matrix()
passergerIDtest1=passergerIDtest1.squeeze()
out1 = net(x1)
prediction1 = torch.max(F.softmax(out1), 1)[1] 
pred_y1 = prediction1.data.numpy().squeeze()
print(passergerIDtest)
print(pred_y1)


datasub = pd.read_csv("E:\\MachineLearning\\pytorch\\gender_submission.csv", usecols=['Survived']) 
y2=datasub.as_matrix().reshape((-1,1))
y2=torch.from_numpy(y2).type(torch.LongTensor).squeeze()
y2 = y2.data.numpy()
print(y2)
accuracy = sum(y2 == pred_y1)/418.  # 预测中有多少和真实值一样
print(accuracy)

datatest_pred = list(zip(passergerIDtest1,pred_y1))
datatest_pred_dataframe = pd.DataFrame(data = datatest_pred ,columns=['PassengerId','Survived'])
print(datatest_pred_dataframe)

datatest_pred_dataframe.to_csv("E:\\MachineLearning\\pytorch\\test_pred.csv", index=False, header=True )
