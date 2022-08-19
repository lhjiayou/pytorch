# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:56:05 2022

@author: huanghao2
"""

'''fashion-mnist的分类实战'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 配置GPU，这里有两种方式
## 方案一：使用os.environ
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可,因为这是cpu版本的，所以需要.cuda()的地方改成.to(device)就可以在无gpu的环境中运行了。
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 256
num_workers = 0   # 对于Windows用户，这里应设置为0，否则会出现多线程错误，不过封装在main中，是可以的
lr = 1e-4
epochs = 20


#数据读入和加载
# 这里同时展示两种方式:
# 下载并使用PyTorch提供的内置数据集
# 从网站下载以csv格式存储的数据，读入并转成预期的格式
# 第一种数据读入方式只适用于常见的数据集，如MNIST，CIFAR10等，PyTorch官方提供了数据下载。这种方式往往适用于快速测试方法（比如测试下某个idea在MNIST数据集上是否有效）
# 第二种数据读入方式需要自己构建Dataset，这对于PyTorch应用于自己的工作中十分重要
# 同时，还需要对数据进行必要的变换，比如说需要将图片统一为一致的大小，以便后续能够输入网络训练；需要将数据格式转为Tensor类，等等。
# 这些变换可以很方便地借助torchvision包来完成，这是PyTorch官方用于图像处理的工具库，上面提到的使用内置数据集的方式也要用到。

## 首先设置数据变换
from torchvision import transforms #其实在数据增强方面很有用
image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),  # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要，就是将数据转化成PIL格式PIL数据读取格式: PIL库读取的图片格式是 H x W x C 格式的
    transforms.Resize(image_size),  #裁减为同一个大小，才能够批量的使用卷积神经
    transforms.ToTensor()
])

## 读取方式一：使用torchvision自带数据集，下载可能需要一段时间
from torchvision import datasets 
train_data = datasets.FashionMNIST(root='./', train=True, download=True, transform=data_transform)  
#那么设置了root之后的数据会保存在哪儿呢？ 下载的时候发现会保存到当前文件夹下./FashionMNIST\raw\
test_data = datasets.FashionMNIST(root='./', train=False, download=True, transform=data_transform)


### 读取方式二：读入csv格式的数据，自行构建Dataset类
# csv数据下载链接：https://www.kaggle.com/zalando-research/fashionmnist
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values  #因为第一列是label
         
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)   #784转化成28*28，H*W*C的numpy数据，还要转化成PIL格式
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)    #如果没有transform，就归一化即可
        label = torch.tensor(label, dtype=torch.long)  #转换成tensor
        return image, label

train_df = pd.read_csv("./fashion-mnist_train.csv")   #train_df.shape= (60000, 785)
test_df = pd.read_csv("./fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)
train_loader = DataLoader(train_data, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers, 
                          drop_last=True)
test_loader = DataLoader(test_data, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         num_workers=num_workers)

#看一个数据
import matplotlib.pyplot as plt
image, label = next(iter(train_loader))   #这个就是读取数据的一个批量的方式，image.shape=torch.Size([256, 1, 28, 28]),label.shape=torch.Size([256])
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")

#完成数据后，构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),   #从(1,28,28) 变成(32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  #这个stride需要些么？其实默认的就是kernel_size,现在变成(32,12,12)
            nn.Dropout(0.3),  #可以学习一下dropout的使用位置
            
            nn.Conv2d(32, 64, 5),  #变成(64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), #变成(64,4,4)
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)  #从(1,28,28) 变成(64,4,4)
        x = x.view(-1, 64*4*4)  #展平一下
        x = self.fc(x)  #10分类的输出
        # x = nn.functional.normalize(x)
        return x

model = Net()
# model = model.cuda()
model=model.to(device)
# model = nn.DataParallel(model).cuda()   # 多卡训练时的写法，之后的课程中会进一步讲解



#设定损失函数
# ?nn.CrossEntropyLoss # 这里方便看一下weighting等策略  weight: Union[torch.Tensor, NoneType] = None,在官方文档中看也可以
criterion = nn.CrossEntropyLoss()


#设定优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)  #这个lr好像就是默认的

#封装的训练与验证函数
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        # data, label = data.cuda(), label.cuda()
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()   #清空梯度
        output = model(data)    #前向计算输出和损失
        loss = criterion(output, label)
        loss.backward()         #反向传播
        optimizer.step()        #模型参数更新
        train_loss += loss.item()*data.size(0)   #加上这个iteration的损失
    train_loss = train_loss/len(train_loader.dataset)  #一个epoch的整个数据集的loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))  #使用%d占位也可以
    return train_loss
def val(epoch):       
    model.eval()
    val_loss = 0
    gt_labels = []  #存储真实的label
    pred_labels = []  #存储预测的label
    with torch.no_grad():
        for data, label in test_loader:
            # data, label = data.cuda(), label.cuda()
            data,label=data.to(device),label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)   #预测的label
            
            gt_labels.append(label.cpu().data.numpy())   #tensor.data.numpy()就获得了tensor的内容的numpy格式
            pred_labels.append(preds.cpu().data.numpy())
            
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)   #将list中的每个长度为batch_size的array拼接成整个array
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)   #整个数据集上的精度
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))    
    return val_loss,acc


train_loss_per_epoch=[]
val_loss_per_epoch=[]
val_acc_per_epoch=[]
for epoch in range(1, epochs+1):
    train_loss=train(epoch)
    val_loss,val_acc=val(epoch)
    
    train_loss_per_epoch.append(train_loss)
    val_loss_per_epoch.append(val_loss)
    val_acc_per_epoch.append(val_acc)
#这只是个简单的实例，因为理论上应该在训练集中8/2划分数据集，得到训练集和验证集，利用验证集反馈确定最佳的模型参数
#然后在整个训练数据集上，用最佳的模型训练，最后就是让这个模型去预测测试集   
    
    

#模型保存,事实上可以只保存模型训练出来的参数，而不需要保存整个model
save_path = "./FahionModel.pkl"
torch.save(model, save_path)





























