# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:37:42 2022

@author: huanghao2
"""


#3.1 对深度学习的整体流程的简单概括
'''机器学习（包括深度学习）流程：数据准备，模型选择，模型训练，然后调整模型不断寻求最佳的验证效果
首先需要对数据进行预处理，其中重要的步骤包括数据格式的统一和必要的数据变换，同时划分训练集和测试集。
接下来选择模型，并设定损失函数和优化方法，以及对应的超参数（当然可以使用sklearn这样的机器学习库中模型自带的损失函数和优化器）。
最后用模型去拟合训练集数据，并在验证集/测试集上计算模型表现'''


'''
但是DL相比较ML存在以下的差异：
1.数据加载层面，dataset和dataloader
由于深度学习所需的样本量很大，一次加载全部数据运行可能会超出内存容量而无法实现；
同时还有批（batch）训练等提高模型表现的策略，需要每次训练读取固定数量的样本送入模型中训练，因此深度学习在数据加载上需要有专门的设计。
2.模型架构需要不断调整，这正是炼丹的本质所在
深度神经网络往往需要“逐层”搭建，或者预先定义好可以实现特定功能的模块，再把这些模块组装起来。
这种“定制化”的模型构建方式能够充分保证模型的灵活性，
3.在损失函数层面，DL一般学术问题上会有更多的自定义损失函数
损失函数和优化器的设定。这部分和经典机器学习的实现是类似的。但由于模型设定的灵活性，
因此损失函数和优化器要能够保证反向传播能够在用户自行定义的模型结构上实现
'''

'''注意的是现在大一点的数据集都是需要GPU的，模型、数据、损失函数、优化器都需要转移到GPU上进行训练
程序默认是在CPU上运行的，因此在代码实现中，需要把模型和数据“放到”GPU上去做运算，
同时还需要保证损失函数和优化器能够在GPU上工作。
如果使用多张GPU进行训练，还需要考虑模型和数据分配、整合的问题。此外，后续计算一些指标还需要把数据“放回”CPU。
这里涉及到了一系列有关于GPU的配置和操作。'''


'''DL的特点:这也正是一般的流程，批的前向计算，梯度清零，计算梯度，反向传播
深度学习中训练和验证过程最大的特点在于读入数据是按批的，
每次读入一个批次的数据，放入GPU中训练，然后将损失函数反向传播回网络最前面的层，同时使用优化器调整网络参数。
这里会涉及到各个模块配合的问题。训练/验证后还需要根据设定好的指标计算模型表现。
'''

#3.2 基本配置
'''常见包：
os、numpy等，用于文件位置管理，numpy主要是可以在创建数据集的时候提供，torch.from_numpy
此外还需要调用PyTorch自身一些模块便于灵活使用，比如
torch、
torch.nn、  各种层
torch.utils.data.Dataset、  工具模块中的数据和数据加载器
torch.utils.data.DataLoader、
torch.optimizr  优化器

比如涉及到表格信息的读入很可能用到pandas，
对于不同的项目可能还需要导入一些更上层的包如cv2等。  对于CV就有很多比较高级的图像处理的包了
如果涉及可视化还会用到matplotlib、seaborn等。
涉及到下游分析和指标计算也常用到sklearn。  甚至对于train_valid_split也是可以使用sklearn的
'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer


'''一般在main文件的开始地方，就设置一下超参数：
batch size               batch_size = 16

初始学习率（初始）        lr = 1e-4，不过一般是1e-3比较堵

训练次数（max_epochs）    max_epochs = 100

GPU配置
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
当然对于我们的torch.cuda.is_available()，返回的仍然是false
'''

#3.3 数据读入
'''PyTorch数据读入是通过Dataset+DataLoader的方式完成的，
Dataset定义好数据的格式和数据变换形式，
DataLoader用iterative的方式不断读入批次数据。'''
'''在自定义dataset的时候，有三个函数，而且下面这三个函数都是特殊的可直接访问的
__init__: 用于向类中传入外部参数，同时定义样本集

__getitem__: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据

__len__: 用于返回数据集的样本数
'''

#eg1，利用ImageFolder来读取分门别类的图像
import torch
from torchvision import datasets
train_data = datasets.ImageFolder(train_path, transform=data_transform)
val_data = datasets.ImageFolder(val_path, transform=data_transform)
'''
注意上面的ImageFolder其实是需要存储的文件位置有一定的要求，也就是说分门别类处理好了的
PyTorch自带的ImageFolder类的用于读取按一定结构存储的图片数据（path对应图片存放的目录，
目录下包含若干子目录，每个子目录对应属于同一个类的图片）

data_transform”可以对图像进行一定的变换，如翻转、裁剪等操作，可自己定义
'''


#eg2：不同于eg1，因为ImageFolder要求图片是分门别类好的，如果所有的图像都在一个文件夹中，其label在一个csv中提供
'''
其中图片存放在一个文件夹，另外有一个csv文件给出了图片名称对应的标签。这种情况下需要自己来定义Dataset类
'''
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.这是图片的位置
            info_csv: path to the csv file containing image indexes 这是图片的index及其标签
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set  数据及划分的txt文件
            transform: optional transform to be applied on a sample.  可选的对图片进行数据增强的
        """
        label_info = pd.read_csv(info_csv)            #这两行还是拿到数据集打开看看更合适
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        #下面这些代码拿到数据集更好理解
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        
        if self.transform is not None:  #数据增强
            image = self.transform(image)
        return image, label

    def __len__(self):  #整个数据集的大小
        return len(self.image_file)
#上面自定义了数据集类，可以进一步实例化之后，用datafolder来读取
train_data=MyDataset(data_dir, info_csv, image_list)
val_data=MyDataset(data_dir, info_csv, image_list)
train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           num_workers=4,   #这个多线程可能会出错
                                           shuffle=True, 
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
#经过上述的数据集构造和数据加载器的设置之后，怎么看看数据的样式呢
'''
PyTorch中的DataLoader的读取可以使用next和iter来完成
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))  #这就会获得一个批量的数据
print(images.shape)  #维度应该是batch，C,H,W？
plt.imshow(images[0].transpose(1,2,0))  #将channel的位置转一下
plt.show()
'''
'''
我之前想看数据的格式都是这样的：
for i,(images,labels) in enumerate(val_loader):
    print(images.shape)  
    plt.imshow(images[0].transpose(1,2,0))  
    plt.show()
    break
加一个break只跑一次
'''


#3.4 模型构建
'''人工智能的第三次浪潮受益于卷积神经网络的出现和BP反向传播算法的实现，随着深度学习的发展，研究人员研究出了许许多多的模型，
PyTorch中神经网络构造一般是基于 Module 类的模型来完成的
这其实也就是class net(nn.Module): 继承这个模块
    '''


#3.4.1 神经网络的构建
'''一个简单的MLP
这里定义的 MLP 类重载了 Module 类的 init 函数和 forward 函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。
'''
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层，也就是隐藏层和输出层
  def __init__(self, **kwargs):  #其实这个参数，可以让模型实例化的时候输入
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)  #继承这个父类
    self.hidden = nn.Linear(784, 256)  #这个位置的超参数是可以实例化的时候输入的
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))  #这个很好理解，就是不断的前向计算即可，至于backward，其实是自动的autograd实现的
    return self.output(o) 

X = torch.rand(2,784)
net = MLP()
print(net)
'''这个和keras的model.summary()很像
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
'''
net(X) #就能够完成一次简单的前向计算，输出的就是forward函数中的return内容

#3.4.2 神经网络中常见的层
#eg1：不含模型参数的层，其实这个就是z-score标准化
import torch
from torch import nn
class MyLayer(nn.Module):
    def __init__(self, **kwargs):   #这儿不需要任何的模型参数
        super(MyLayer, self).__init__(**kwargs)
    def forward(self, x):
        # return x - x.mean()    #前向计算就是减去均值
        return (x-x.mean())/x.std()
layer = MyLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))


#eg2,含模型参数的层
'''
Parameter 类其实是 Tensor 的子类，如果一 个 Tensor 是 Parameter ，那么它会自动被添加到模型的参数列表里。
所以在定义含模型参数的层时，我们应该将参数定义成 Parameter ，
除了直接定义成 Parameter 类外，还可以使用ParameterList 和 ParameterDict 分别定义参数的列表和字典
'''
#eg2-1使用ParameterList来定义参数列表
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])  #定义参数列表
        self.params.append(nn.Parameter(torch.randn(4, 1)))  #这个好像是输出层的参数

    def forward(self, x):
        for i in range(len(self.params)):   #好像是三层的隐藏层加上最后一层的输出层
            x = torch.mm(x, self.params[i])
        return x
net = MyListDense()
print(net)
'''
MyListDense(
  (params): ParameterList(
      (0): Parameter containing: [torch.FloatTensor of size 4x4]
      (1): Parameter containing: [torch.FloatTensor of size 4x4]
      (2): Parameter containing: [torch.FloatTensor of size 4x4]
      (3): Parameter containing: [torch.FloatTensor of size 4x1]
  )
)
'''
x=torch.rand(5,4)  #输入的是(5,4)
y=net(x)           #那么现在输出的就是(5,1)

#eg2-2使用 ParameterDict来定义参数字典
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 4))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增，因为list可以使用append，对于字典用的是update？

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)
'''
MyDictDense(
  (params): ParameterDict(
      (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
      (linear2): Parameter containing: [torch.FloatTensor of size 4x4]
      (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
  )
)
'''
x=torch.rand(5,4)  
y=net(x)   #因为选择了choice，那么只会使用第一层

#eg2-3对上面的MyDictDense改写一下：
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 4))
        })
        # self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增，因为list可以使用append，对于字典用的是update？
        self.params['linear3']=nn.Parameter(torch.randn(4, 2)) #对于字典不使用update，直接用字典新增键值对的方式也可以

    def forward(self, x, ):
        for key,value in self.params.items():  #好像是三层的隐藏层加上最后一层的输出层
            x = torch.mm(x, value)
        return x
net = MyDictDense()
print(net)
x=torch.rand(5,4)  
y=net(x) 

####二维卷积层，如果手工实现，没有考虑padding的时候
import torch
from torch import nn
def corr2d(X, K): 
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):  #这两层for循环应该可以使用列表解析实现
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))  #提供正态分布的初始化参数
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias #将输入x和初始化的权重按照上面定义的卷积运算进行前向计算
    
####二维卷积层，方形卷积核
import torch
from torch import nn
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数
    X = X.view((1, 1) + X.shape)  #那么现在就是将输入数据从(8,8)的H*W变成(1,1,8,8)的B*C*H*W
    Y = conv2d(X)   #这个位置其实是调用的现有的二维卷积层
    return Y.view(Y.shape[2:]) # 排除不关心的前两维:批量和通道，也就是B*C*H*W再变成H*W输出
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)#,padding=1) #实例化一个二维卷积层，kernel-szie=3的时候padding设置为1
X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape    #torch.Size([8, 8]),但是没有padding之后就变成了torch.Size([6, 6])

####二维卷积层，非方形卷积核
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))  #padding=kernel_size-1/2
comp_conv2d(conv2d, X).shape   #torch.Size([8, 8])

####二维卷积核，而且stride不是1
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape




####池化层，手工实现
import torch
from torch import nn
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):  #这两层for循环，以及下面的if-else其实可以用numpy的高效实现
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float)
print(X)
pool2d(X, (2, 2))  #池化窗口大小为2
pool2d(X, (2, 2), 'avg')    
    

#3.4.3 几个模型的pytorch实现
#eg1，lenet
import torch 
import torch.nn as nn
import torch.nn.functional as F   #主要是池化要用的
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  #32-5+1=28
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) #14-5+1=10
        self.fc1=nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.fc3=nn.Linear(in_features=84, out_features=10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.flat(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    def flat(self,x):
        # batch=x.size()[0]   #其实和X.shape[0]   是一样的，shape在torch和numpy中都是可以使用的
        feature=x.size()[1:]
        dim=np.cumprod(feature)[-1]   #没有使用for循环
        return dim
net =LeNet()
print(net)  
#上面就是第一次简单实现NN，而且我们只需要定义 forward 函数，backward函数会在使用autograd时自动定义，backward函数用来计算导数。
#至于怎么利用梯度下降实现可训练权重参数的更新，也是optimizer.step()做的事情了
params = list(net.parameters())  #10个parameter组成的list
print(len(params))
print(params[0].size())  # conv1的权重  
  
  
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)   

net.zero_grad()
out.backward(torch.randn(1, 10))  #这其实是因为out不是一个标量，必须输入给一个参数
    
'''
torch.nn只支持小批量处理 (mini-batches）。整个 torch.nn 包只支持小批量样本的输入，不支持单个样本的输入。
比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width 
如果是一个单独的样本，只需要使用input.unsqueeze(0) 来添加一个“假的”批大小维度，这个0指的就是什么位置增加这个1
另外
input = torch.randn(1, 32, 32)
input=input.view((1,)+input.shape)也能够增加一个维度，但是在中间增加并不是很方便
'''
# x=torch.randn(1,1,224,224)
# conv1=nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11,stride=4)  
# conv1(x).shape   torch.Size([1, 96, 54, 54])
#eg2.AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
            )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc=nn.Sequential(  
            nn.Linear(256*5*5, 4096),    #在这个地方的256*5*5其实是必须输入的，不然没有办法初始化这个NN
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
            )
    def forward(self, img):    #img的shape是(batch,1,224,224)
        feature = self.conv(img)  #经过卷积层之后维度是(batch,256,5,5)
        output = self.fc(feature.view(img.shape[0], -1))   #flatten变成(batch,256*5*5)之后，不断地降维变成(batch,10)
        # return output
        return output

    
net = AlexNet()
print(net)    
x=torch.randn(1,1,224,224)   
net(x).shape    


#3.5 模型初始化
#3.5.1 torch.nn.init的内容
'''在深度学习模型的训练中，权重的初始值极为重要。一个好的权重值，会使模型收敛速度提高，使模型准确率更精确。
为了利于训练和减少收敛时间，我们需要对模型进行合理的初始化。PyTorch也在torch.nn.init中为我们提供了常用的初始化方法。
1 . torch.nn.init.uniform_(tensor, a=0.0, b=1.0) 
2 . torch.nn.init.normal_(tensor, mean=0.0, std=1.0) 
3 . torch.nn.init.constant_(tensor, val) 
4 . torch.nn.init.ones_(tensor) 
5 . torch.nn.init.zeros_(tensor) 
6 . torch.nn.init.eye_(tensor) 
7 . torch.nn.init.dirac_(tensor, groups=1) 
8 . torch.nn.init.xavier_uniform_(tensor, gain=1.0) 
9 . torch.nn.init.xavier_normal_(tensor, gain=1.0) 
10 . torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan__in', nonlinearity='leaky_relu') 
11 . torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
12 . torch.nn.init.orthogonal_(tensor, gain=1) 
13 . torch.nn.init.sparse_(tensor, sparsity, std=0.01) 
14 . torch.nn.init.calculate_gain(nonlinearity, param=None)
'''

#3.5.2torch.nn.init的使用
#通常使用isinstance来进行判断模块（回顾3.4模型构建）属于什么类型。
conv = nn.Conv2d(1,3,3)
linear = nn.Linear(10,1)

isinstance(conv,nn.Conv2d)
isinstance(linear,nn.Conv2d)

# 查看随机初始化的conv参数
conv.weight.data
conv.bias.data
# 查看linear的参数
linear.weight.data

# 对conv进行kaiming初始化
torch.nn.init.kaiming_normal_(conv.weight.data)   #对这个权重参数的数值原地修改
conv.weight.data
# 对linear进行常数初始化
torch.nn.init.constant_(linear.weight.data,0.3)   #常数初始化
linear.weight.data


#3.5.3 初始化函数的封装
'''人们常常将各种初始化方法定义为一个initialize_weights()的函数并在模型初始后进行使用。
例如下面函数就是遍历当前模型的每一层，然后判断各层属于什么类型，然后根据不同类型层，设定不同的权值初始化方法'''
def initialize_weights(self):
	for m in self.modules():  #每个层
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置，有的话用常数初始化
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
        #判断是否属于线形层
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
        #判断是否属于BN层
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()	
# 模型的定义
class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Conv2d(1,1,3)
    self.act = nn.ReLU()
    self.output = nn.Linear(10,1)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)

#没有初始化之前
mlp = MLP()
print(list(mlp.parameters()))
print("-------初始化-------")
#初始化之后
initialize_weights(mlp)
print(list(mlp.parameters()))  #但是可见除了常数初始化之外，其它的和教程中的并不相等，怎么保证可复现性需要设置



#3.6 损失函数，特别是自定义损失函数非常的关键
#3.6.1 二分类交叉熵损失函数
torch.nn.BCELoss(weight=None,    #weight:每个类别的loss设置权值，例如不平衡的分类问题，可能会使用
                 size_average=None, #size_average:数据为bool，为True时，返回的loss为平均值；为False时，返回的各样本的loss之和，要不要平均的问题
                 reduce=None,   #reduce:数据类型为bool，为True时，loss的返回是标量。
                 reduction='mean')  #这个和size_average有什么区别呢
'''问题1，reduce，size-average以及reduction之间的关系'''
#可以参考连接https://blog.csdn.net/u013548568/article/details/81532605
#在reduce=true而且size_average=true的时候，输出的就是batch的loss的均值
#在reduce=true而且size_average=false的时候，输出的就是batch的loss的和
#在reduce=false的时候，  假设输入和target的大小分别是NxCxWxH，那么一旦reduce设置为False，loss的大小为NxCxWxH，返回每一个元素的loss
#reduction代表了上面的reduce和size_average双重含义，这也是文档里为什么说reduce和size_average要被Deprecated 的原因
#所以事实上我们常见的就是设置reduction=mean即可
'''问题2，这个输入要求是概率，也就是说这个损失函数不会自动计算softmax'''
#对于进入交叉熵函数的input为概率分布的形式。一般来说，input为sigmoid激活层的输出，或者softmax的输出
#也就是说BCELoss并没有自动进行softmax，前面需要使用softmax激活转化成概率形式


m = nn.Sigmoid()  #转化成概率之后才能输入BCE种
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
output.backward()
print('BCELoss损失函数的计算结果为',output)


# 3.6.2 交叉熵损失函数
torch.nn.CrossEntropyLoss(weight=None, 
                          size_average=None, 
                          ignore_index=-100,   #ignore_index:忽略某个类的损失函数。还有这个功能？
                          reduce=None, 
                          reduction='mean')
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)            #标签预测值
target = torch.empty(3, dtype=torch.long).random_(5)     #标签真实值
output = loss(input, target)                             #内部会计算softmax，因此fc的输出其实不需要进行激活
output.backward()
print(output)



#3.6.3 L1损失函数
'''功能是：计算输出y和真实标签target之间的差值的绝对值，也就是L1范数
reduction参数决定了计算模式。有三种计算模式可选：
none：逐个元素计算。   如果选择none，那么返回的结果是和输入元素相同尺寸的。
sum：所有元素求和，返回标量。 
mean：加权平均，返回标量。   默认的
'''
torch.nn.L1Loss(size_average=None, 
                reduce=None, 
                reduction='mean')

loss = nn.L1Loss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
print('L1损失函数的计算结果为',output)

# 3.6.4 MSE损失函数
'''功能： 计算输出y和真实标签target之差的平方。
和L1Loss一样，MSELoss损失函数中，reduction参数决定了计算模式。
有三种计算模式可选：none：逐个元素计算。 sum：所有元素求和，返回标量。默认计算方式是求平均。'''
torch.nn.MSELoss(size_average=None, 
                 reduce=None, 
                 reduction='mean')
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
print('MSE损失函数的计算结果为',output)

#3.6.5 平滑L1 (Smooth L1)损失函数，其实可见在偏差较小的时候用的是L2范数，偏差较大的时候用的是L1范数
'''功能： L1的平滑输出，其功能是减轻离群点带来的影响，这个其实还是很关键的，具体计算公式可以看教程内容
reduction参数决定了计算模式。有三种计算模式可选：none：逐个元素计算。 sum：所有元素求和，返回标量。默认计算方式是求平均。'''
torch.nn.SmoothL1Loss(size_average=None, 
                      reduce=None, 
                      reduction='mean', 
                      beta=1.0)   #这个beta就是对离群点的容忍程度
loss = nn.SmoothL1Loss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
print('SmoothL1Loss损失函数的计算结果为',output)


#平滑L1与L1的对比
# 这里我们通过可视化两种损失函数曲线来对比平滑L1和L1两种损失函数的区别。
inputs = torch.linspace(-10, 10, steps=5000)
target = torch.zeros_like(inputs)  #全是0，就使得inputs -target=inputs 

loss_f_smooth = nn.SmoothL1Loss(reduction='none')   #为什么使用的reduction=none，这样才能够保证每个点都有一个损失函数，否则就是输出一个标量了
loss_smooth = loss_f_smooth(inputs, target)  #loss_smooth.shape=torch.Size([5000])
loss_f_l1 = nn.L1Loss(reduction='none')
loss_l1 = loss_f_l1(inputs,target)

import matplotlib.pyplot as plt
plt.plot(inputs.numpy(), loss_smooth.numpy(), label='Smooth L1 Loss')
plt.plot(inputs.numpy(), loss_l1, label='L1 loss')
plt.xlabel('x_i - y_i')
plt.ylabel('loss value')
plt.legend()
plt.grid()
plt.show()
#可以看出，对于smoothL1来说，在 0 这个尖端处，过渡更为平滑。不过其实正好出现0的可能性不大，增加了光滑就使得梯度的计算成本更大了

# 3.6.6 目标泊松分布的负对数似然损失  不是很常见

#3.6.7 KL散度
'''功能： 计算KL散度，也就是计算相对熵。用于连续分布的距离度量，并且对离散采用的连续输出空间分布进行回归通常很有用。
主要参数:
reduction：计算模式，可为 none/sum/mean/batchmean。
none：逐个元素计算。
sum：所有元素求和，返回标量。
mean：加权平均，返回标量。
batchmean：batchsize 维度求平均值。
'mean' will be changed to behave the same as 'batchmean' in the next major release.
'''
torch.nn.KLDivLoss(size_average=None, 
                   reduce=None, 
                   reduction='mean', 
                   log_target=False)
inputs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2]], dtype=torch.float)
loss = nn.KLDivLoss()
output = loss(inputs,target)
print('KLDivLoss损失函数的计算结果为',output)


# 3.6.8 MarginRankingLoss
#排序损失函数；输入的是x1,x2,y， 其中x1排在x2的前面的时候，标签y为+1，而x1排在x2的后面的时候，标签y为-1
'''功能： 计算两个向量之间的相似度，用于排序任务。该方法用于计算两组数据之间的差异。

主要参数:

margin：边界值， 与 之间的差异值。

reduction：计算模式，可为 none/sum/mean。'''
loss = nn.MarginRankingLoss()
input1 = torch.randn(3, requires_grad=True)  #tensor([-0.2874,  0.4411,  0.1080], requires_grad=True)
input2 = torch.randn(3, requires_grad=True)  # tensor([-1.1801,  0.8447, -0.0296], requires_grad=True)
target = torch.randn(3).sign()  # tensor([1., 1., 1.])
output = loss(input1, input2, target)
output.backward()

print('MarginRankingLoss损失函数的计算结果为',output)



#3.6.9 多标签边界损失函数
#功能： 对于多标签分类问题计算损失函数。
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')

#3.6.10 二分类损失函数
#功能： 计算二分类的 logistic 损失。
torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')

#3.6.11 多分类的折页损失,hinge损失函数？
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')


#3.6.12  三元组损失
'''功能： 计算三元组损失。

三元组: 这是一种数据的存储或者使用格式。<实体1，关系，实体2>。在项目中，也可以表示为< anchor, positive examples , negative examples>

在这个损失函数中，我们希望去anchor的距离更接近positive examples，而远离negative examples'''
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')



#3.6.13 HingEmbeddingLoss
# 功能： 对输出的embedding结果做Hing损失计算
torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')


#3.6.14 余弦相似度,这个可以关注下

'''功能： 对两个向量做余弦相似度
主要参数:reduction：计算模式，可为 none/sum/mean。
margin：可取值[-1,1] ，推荐为[0,0.5] 。'''
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
'''基本原理是;对于两个向量，做余弦相似度。将余弦相似度作为一个距离的计算方式，如果两个向量的距离近，则损失函数值小，反之亦然。'''
loss_f = nn.CosineEmbeddingLoss()
inputs_1 = torch.tensor([[0.3, 0.5, 0.7], [0.3, 0.5, 0.7]],requires_grad=True)
inputs_2 = torch.tensor([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5]],requires_grad=True)
target = torch.tensor([1, -1], dtype=torch.float)

output = loss_f(inputs_1,inputs_2,target)
print('CosineEmbeddingLoss损失函数的计算结果为',output)


# [np.linalg.norm(inputs_1.detach().numpy()[i],2) for i in range(inputs_1.shape[0])] #使用了.detach().numpy()之后,inputs-1事实上还是有梯度的

# import numpy as np
def cos(x,y,label,margin=0,reduction='mean'):
    '''下面分别是numpy写法和torch写法'''
    # x,y,label=inputs_1,inputs_2,target
    #经过下面的运算后，虽然x，y，label都还是有grad的，但是其他的都是数组
    # fenzi=np.array([x.detach().numpy()[i].dot(y.detach().numpy()[i]) for i in range(x.shape[0])])
    # fenmu_x=np.array([np.linalg.norm(x.detach().numpy()[i],2) for i in range(x.shape[0])])
    # fenmu_y=np.array([np.linalg.norm(y.detach().numpy()[i],2) for i in range(y.shape[0])])
    # cos=fenzi/(fenmu_x*fenmu_y)  
    # if reduction=='mean':
    #     loss=torch.mean(torch.Tensor([1-cos[i] if label[i]==1 else max([0,cos[i]-margin]) for i in range(x.shape[0])]))
    # elif reduction=='sum':
    #     loss=torch.sum(torch.Tensor([1-cos[i] if label.detach().numpy()[i]==1 else max([0,cos[i]-margin]) for i in range(x.shape[0])]))
    # else:
    #     loss=torch.Tensor([1-cos[i] if label.detach().numpy()[i]==1 else max([0,cos[i]-margin]) for i in range(x.shape[0])])
    # return loss    
    fenzi=[torch.dot(x[i],y[i]) for i in range(x.shape[0])]
    fenmu_x=[torch.norm(x[i],2) for i in range(x.shape[0])]
    fenmu_y=[torch.norm(y[i],2) for i in range(y.shape[0])]
    cos=[fenzi[i]/(fenmu_x[i]*fenmu_y[i]) for i in range(x.shape[0])] 
    loss=torch.zeros(x.shape[0])    
    for i in range(x.shape[0]):
        if label[i]==1:
            loss[i]=1-cos[i]
        else:
            loss[i]=max([0,cos[i]-margin])
    if reduction=='mean':
        loss=torch.mean(loss)
    elif reduction=='sum':
        loss=torch.sum(loss)
    else:
        loss=loss
    return loss
output =cos(inputs_1,inputs_2,target,margin=0,reduction='mean')
print('CosineEmbeddingLoss损失函数的计算结果为',output)

        

#3.6.15 CTC损失函数
#https://www.jianshu.com/p/0cca89f64987  写的很详细
'''
功能： 用于解决时序类数据的分类
计算连续时间序列和目标序列之间的损失。CTCLoss对输入和目标的可能排列的概率进行求和，产生一个损失值，这个损失值对每个输入节点来说是可分的。输入与目标的对齐方式被假定为 "多对一"，这就限制了目标序列的长度，使其必须是≤输入长度。
主要参数:
reduction：计算模式，可为 none/sum/mean。
blank：blank label。
zero_infinity：无穷大的值或梯度值为
'''
# Target are to be padded，输出的标签是需要进行padding的
import torch
import torch.nn as nn
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()


# Target are to be un-padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()  #log_softmax其实就是log加上sofmtmax，2指的是axis=2

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()
print('CTCLoss损失函数的计算结果为',loss.item())  
'''detach的作用：https://www.cnblogs.com/ljf-0/p/14015601.html
https://blog.csdn.net/qq_31244453/article/details/112473947  
https://blog.csdn.net/qq_27825451/article/details/95498211?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control写的非常的好
a = torch.tensor([1, 2, 3.], requires_grad=True)
out = a.sigmoid()
print(out)  #tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward0>)
c=out.data
print(c)  #tensor([0.7311, 0.8808, 0.9526])
c[0]=1
print(c)  #tensor([1.0000, 0.8808, 0.9526])
print(out)  #tensor([1.0000, 0.8808, 0.9526], grad_fn=<SigmoidBackward0>)，可以发现其实此时的out的data也修改了
out.sum().backward()  #现在是可以进行反向传播的

a = torch.tensor([1, 2, 3.], requires_grad=True)
out = a.sigmoid()
c = out.detach()
print(c)   #tensor([0.7311, 0.8808, 0.9526])
c[0]=1
print(c)  #tensor([1.0000, 0.8808, 0.9526])
print(out)  #tensor([1.0000, 0.8808, 0.9526], grad_fn=<SigmoidBackward0>)，out的数值也发何时能了变化，但是仍然存在梯度
out.sum().backward()   #这个时候就会报错了
'''

#3.7 训练和评估
model.train()   # 训练状态
model.eval()   # 验证/测试状态
'''model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度
with torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播。'''

'''在3.2节说了设置GPU的两种方法：
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

'''

#训练的流程为:
for data, label in train_loader:
#或者for i,(data,label) in enumerate(train_loader):    
    data, label = data.cuda(), label.cuda() #将数据转到gpu上
    optimizer.zero_grad() #第一步，梯度清零
    output = model(data) #第二步，前向计算
    loss = criterion(output, label)
    loss.backward()  #第三步，反向传播
    optimizer.step() #参数更新
    
#验证/测试与训练的差别;
# 验证/测试的流程基本与训练过程一致，不同点在于：
# 1.需要预先设置torch.no_grad，以及将model调至eval模式,有些时候model.eval()其实就可以了
# 2.不需要将优化器的梯度置零
# 3.不需要将loss反向回传到网络  验证计算的loss只是为了显示模型的性能，并不是为了更新模型参数的
# 4.不需要更新optimizer

#将上面的训练流程封装成函数;
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(label, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)   #这是因为上面的criterion计算的是默认的mean，其实在criterion实例化的时候reduction='sum’就可以
    train_loss = train_loss/len(train_loader.dataset) #计算的是整个数据集上一次epoch的loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss)) 
#将上面的验证/测试流程封装成函数;
def val(epoch):       
    model.eval()   #要求不进行方向传播，另外对dropout和BN层进行设置
    val_loss = 0
    with torch.no_grad():  #完全不计算梯度，进一步加快计算速度
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)   #这是分类问题常用的
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0) 
            running_accu += torch.sum(preds == label.data)  #同样是分类的精度
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))

#3.8 可视化
'''分类的ROC曲线，卷积网络中的卷积核，以及训练/验证过程的损失函数曲线
还会对卷积核进行可视化么？一般不就是loss曲线，acc曲线么
'''    


#3.9 优化器optimizer
#3.9.1 各种优化器，在库torch.optim中
'''torch.optim.ASGD

torch.optim.Adadelta

torch.optim.Adagrad

torch.optim.Adam   最常用

torch.optim.AdamW

torch.optim.Adamax

torch.optim.LBFGS

torch.optim.RMSprop  这个好像性能也还是很不错的

torch.optim.Rprop

torch.optim.SGD

torch.optim.SparseAdam'''
'''
一些最常见的方法：
zero_grad()：清空所管理参数的梯度，PyTorch的特性是张量的梯度不自动清零，因此每次反向传播后都需要清空梯度。每个epoch训练的时候都需要
step()：执行一步梯度更新，参数更新，就实现了利用梯度的SGD下降
add_param_group()：添加参数组，没见过，好像不是很常见
load_state_dict() ：加载状态参数字典，可以用来进行模型的断点续训练，继续上次的参数进行训练，
例如在模型保存的时候，只想保存参数以节省空间，在重新加载的时候，实例化一个新model之后再加载这个参数就能够用于新数据的预测了
state_dict()：获取优化器当前状态信息字典
'''

#3.9.2和3.9.3 可以选择性看看不是很重要
#但是3.9.3节给网络的不同的层赋予不同的优化器参数，这个还是有点意思
from torch import optim
from torchvision.models import resnet18
net = resnet18()
print(net) #可以看见是conv1,bn1,relu,maxpool 然后加上layer1,layer2,layer3,layer4(这些都是sequential的),最'后是avgpool，加上fc
'''net.layer1输出的就是
Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)'''
'''
net.fc 输出 Linear(in_features=512, out_features=1000, bias=True)
'''
optimizer = optim.SGD([
    {'params':net.fc.parameters()},#fc的lr使用默认的1e-5
    {'params':net.layer4[0].conv1.parameters(),'lr':1e-2}],lr=1e-5) #net.layer4[0].conv1的参数将使用1e-2进行优化

#3.9.4 实验，一般而言，选择adam的比较多，在adam之前，其实rmsprop也还是很不错的
# #数据生成
# a = torch.linspace(-1, 1, 1000)
# x = torch.unsqueeze(a, dim=1)   #通过dim=1增加一个维度1，变成[1000,1]
# y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))  #维度也是[1000,1]
# import matplotlib.pyplot as plt
# plt.scatter(x,y)


# #创建dataset和dataloader
# from torch.utils.data import DataLoader,Dataset
# train_set=Dataset(x,y)
# train_loader=DataLoader(train_set,
#                         batch_size=200,
#                         shuffle=True,
#                         num_workers=0)
# #创建模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(1, 20)  #特征维度就是从1-20-1
#         self.predict = nn.Linear(20, 1)

#     def forward(self, x):
#         x = self.hidden(x)
#         x = F.relu(x)
#         x = self.predict(x) #标量输出，不需要使用激活函数
#         return x
# #创建损失
# loss=torch.nn.MSELoss()
# #模型训练
# epoches=200
# optim_list=[torch.optim.SGD,
#             torch.optim.ASGD,
#             torch.optim.SGD,  #动量法也是SGD的一种
#             torch.optim.Adadelta
#             ]
# for i,(xx,yy) in enumerate(train_loader):
#     optim




























































