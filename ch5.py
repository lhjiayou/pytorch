# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:45:32 2022

@author: huanghao2
"""
import torch
import torch.nn as nn
#5.1 PyTorch模型定义的方式

#5.1.1 基于nn.Module，我们可以通过Sequential，ModuleList和ModuleDict三种方式定义PyTorch模型。
#虽然我之前都是按照sequential进行构建的，有些很简单的model，其实都没写成block，就是逐层写出来的

#5.1.2 Sequential  对应模块为nn.Sequential()，这是创建模型的最简单的方法
class MySequential(nn.Module):
    '''相当于手动完成了nn.Sequential()，其实在pytorch的官方文档中应该可以查询到相关的代码实现'''
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict 
            for key, module in args[0].items():
                self.add_module(key, module)  
                # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)  #添加模块
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)  #不断地前向计算
        return input
#下面来看下如何使用Sequential来定义模型。只需要将模型的层按序排列起来即可，根据层名的不同，排列的时候有两种方式：
#方式一，直接排列，几乎都是这种方式
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
net[-1]   #竟然也是可以的
net.append(nn.Linear(256, 10))  #但是这个会报错

print(net)
#第二种，使用ordereddict，也就是model的有序排列的字典
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),   #这种对每一层其实都有设置name
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)
'''这种顺序模型的特点：
使用Sequential定义模型的好处在于简单、易读，同时使用Sequential定义的模型不需要再写forward，因为顺序已经定义好了。
但使用Sequential也会使得模型定义丧失灵活性，比如需要在模型中间加入一个外部输入时就不适合用Sequential的方式实现。使用时需根据实际需求加以选择。
这种其实和keras中的add_layer很像'''
#问？上面定义的mysequentional能不能使用？
net3 = MySequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),   #这种对每一层其实都有设置name
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net3)
'''也能够输出：
MySequential(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)'''





#5.1.3 ModuleList  对应模块为nn.ModuleList()。
'''ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作。
同时，子模块或层的权重也会自动添加到网络中来。'''
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
net[-1]  #获取一层非常的方便
print(net[-1])  # 类似List的索引访问，这个很有意思，虽然在功能上和上面的nn.sequential差不多，但是这种要想获取某一层可能比较方便
print(net)
'''
要特别注意的是，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起。
ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，
需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用for循环即可完成：
'''
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...  #也就是在模型初始化的时候提供的model的list
    ...
    
  def forward(self, x):
    for layer in self.modulelist:  #然后在forward中前向计算，那么这种方式其实比第一种麻烦
      x = layer(x)
    return x

# 5.1.4 ModuleDict  对应模块为nn.ModuleDict()。
# ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称。
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 可见这种添加方式，就是词典新增键值对的方式
print(net['linear']) # 访问可以直接指定key，前面的可能是隐式的index
print(net.output)  #这也是一种访问方式
print(net)

#5.1.5 三种方法的比较与适用场景
# Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写__init__和forward；
#但是可能需要重复写，例如10层的fc，而且完全相同，那么就需要写10次

# ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；
# 当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。
#这种的好处就是完全相同的层，或者比较完全相同的模块，可以避免写很长，但是这种可能会让模型的架构不是很清晰


#5.2 利用layer组成模块，再用模块搭建整个完整模型
'''
对于大部分模型结构（比如ResNet、DenseNet等），我们仔细观察就会发现，虽然模型有很多层， 
但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，
如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，
这样将会极大便利模型构建的过程。其实也就是代码重复利用
实现的流程是：将简单层构建成具有特定功能的模型块，再利用模型块构建复杂网络
'''

#5.2.1 U-Net简介
#5.2.2 模块结构分析
'''
不难发现U-Net模型具有非常好的对称性。
模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接；
同时位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”。也就是残差连接的部分
此外还有输入和输出处理等其他组成部分。由于模型的形状非常像英文字母的“U”，因此被命名为“U-Net”。

组成U-Net的模型块主要有如下几个部分：

1）每个子块内部的两次卷积（Double Convolution），这已经是最小的子块的内容了

2）左侧模型块之间的下采样连接，即最大池化（Max pooling），实现维度降低

3）右侧模型块之间的上采样连接（Up sampling），实现维度上升

4）输出层的处理，这个应该是需要展平+fc

根据功能我们将其命名为：DoubleConv, Down, Up, OutConv。

除模型块外，还有模型块之间的横向连接，
输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现。
'''


#5.2.3 模型块的实现
'''
在使用PyTorch实现U-Net模型时，我们不必把每一层按序排列显式写出，这样太麻烦且不宜读，
一种比较好的方法是先定义好模型块，再定义模型块之间的连接顺序和计算方式。
就好比装配零件一样，我们先装配好一些基础的部件，之后再用这些可以复用的部件得到整个装配体。
'''
#对比图上，好像发现5.2.1节图中卷积的时候好像没有使用padding
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    这就是每个子块的两次卷积，conv-BN-RELU，甚至还可以进一步简化
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        #因为会多次调用，所以初始化有着不同的输入输出通道以及中间通道
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  #这个可能后面再看看是什么情况，有存在通道数的多次变化么？
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3,  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
# x=torch.rand(1,1,572,572)
# conv=DoubleConv(1,64)
# y=conv(x)
# y.shape   torch.Size([1, 64, 572, 572])因为存在padding，所以长宽没有发生变化


class Down(nn.Module):
    """Downscaling with maxpool then double conv
    先下采样，再双卷积，是用顺序模型写的
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(    #这就是上面介绍的第一种模型定义方式
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)  #直接实例化上面的双卷积
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv
    先上采样，再双卷积，这一块比down复杂一点，因为存在skip connection
    """

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #如果是双线性的，使用正常卷积来减少通道的数量
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            #否则，使用ConvTranspose2d莱士心维度的上升
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)   #也是直接实例化上面的双卷积

    def forward(self, x1, x2):
        '''输入维度为：
        x1=torch.rand(1,1024,28,28)         x2=torch.rand(1,512,64,64)
        这儿需要实现残差连接，因此其实上面的padding=1可以避免这儿出现维度不一致的问题
        '''
        
        x1 = self.up(x1)    #torch.Size([1, 512, 56, 56])             
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  #64-56=8
        diffX = x2.size()[3] - x1.size()[3]  #64-56=8
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,  #在上下左右都添加这么多的0，torch.Size([1, 512, 64, 64])
                        diffY // 2, diffY - diffY // 2])
        #这一块实现了什么呢？
        # x1=torch.rand(1,1,3,3)
        # tensor([[[[0.5659, 0.6086, 0.1622],
        #           [0.2699, 0.3401, 0.1899],
        #           [0.8655, 0.4584, 0.6797]]]])
        # x1 = F.pad(x1, [4,4,4,4])  变成torch.Size([1, 1, 11, 11])，那就是在上下左右都添加这么多的0么
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)   #拼接之后， [1, 512, 64, 64]  1,512,64,64 -变成[1,1024,64,64]
        return self.conv(x)  #然后再卷积，变成{1,512,60,60]，其中这个512是初始化指定的
# upnet=Up(1024,512)
# x1=torch.rand(1,1024,28,28)
# x2=torch.rand(1,512,64,64)
# y=upnet(x1,x2)
# y.shape

    
class OutConv(nn.Module):
    '''
    输出卷积，关键是kernel=1，其实相当于实现跨通道的信息融合
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#利用上述四个部分，可以搭建整个完整的U-Net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels    #这是输入的通道数
        self.n_classes = n_classes     #输出的好像就是2
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64) 
        self.down1 = Down(64, 128) 
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)   #输入1*572*572
        # return x1          torch.Size([1, 64, 572, 572]),就是输入部分的双卷积，channel变成64
        x2 = self.down1(x1)
        # return x2          torch.Size([1, 128, 286, 286]) 先下采样，572//2=286,然后channel变成128   
        x3 = self.down2(x2)
        # return x3          torch.Size([1, 256, 143, 143]) 先下采样，286//2=143，然后channel变成256
        x4 = self.down3(x3)
        # return x4            torch.Size([1, 512, 71, 71]) 先下采样，143//2=71，然后channel变成512
        x5 = self.down4(x4)
        # return x5             #torch.Size([1, 1024, 35, 35]) 先下采样 71//2=35，然后channel变成1024
        
        #此时已经实现了到达了中间，图中现在是1024*28*28，主要是因为没有padding，可以试试看
        #在双卷积模块中，将padding=1注释了之后，确实出现的是torch.Size([1, 1024, 28, 28])
        
        x = self.up1(x5, x4)    #torch.Size([1, 512, 60, 60])       
        x = self.up2(x, x3)     #torch.Size([1, 256, 132, 132])        
        x = self.up3(x, x2)     # torch.Size([1, 128, 276, 276])        
        x = self.up4(x, x1)     # torch.Size([1, 64, 564, 564])
        logits = self.outc(x)   #torch.Size([1, 2, 564, 564])
        return logits
# net=UNet(n_channels=1, n_classes=2)
# x=torch.rand(1,1,572,572)
# y=net(x)
# y.shape
'''可见整理模型是一个inc，四个down，四个up，一个outc组成的，而且每个'''



#5.3 使用pytorch进行模型修改
'''有越来越多的开源模型可以供我们使用，很多时候我们也不必从头开始构建模型。因此，掌握如何修改PyTorch模型就显得尤为重要。
特别是很多开源模型可以直接魔改

如何在已有模型的基础上：
修改模型若干层
添加额外输入
添加额外输出
'''

#5.3.1 修改模型层
#以pytorch官方视觉库torchvision预定义好的模型ResNet50为例，探索如何修改模型的某一层或者某几层
import torchvision.models as models
net = models.resnet50()
print(net)
'''可见模型的架构包括：
conv1-BN1-RELU-maxpool-layer1/2/3/4-avgpool-fc
layer1/2/3/4其实 分别是3、4、6、3个Bottleneck组成的，
每个Bottleneck都是conv-bn-conv-bn-conv-bn-relu组成的
为了适配ImageNet预训练的权重，因此最后全连接层（fc）的输出节点数是1000。
'''
#现在的需求是修改这个模型使得进行10分类任务
from collections import OrderedDict  
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
#上面的写法其实就是5.1.2节中的sequential模型的两种方式之一（第一种直接排列，每层的name就是
#整数编码，第二层使用的是有序字典，因此每成都会有名字    
net.fc = classifier  #这也太简单了，直接用赋值的方式就把层给修改了
print(net)    


#5.3.2 添加外部输入
'''有时候在模型训练中，除了已有模型的输入之外，还需要输入额外的信息。
比如在CNN网络中，我们除了输入图像，还需要同时输入图像对应的其他信息，
这时候就需要在已有的CNN网络中添加额外的输入变量。

基本思路是：将原模型添加输入位置前的部分作为一个整体，
同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改。'''

'''我们以torchvision的resnet50模型为基础，任务还是10分类任务。
不同点在于，我们希望利用已有的模型结构，
在倒数第二层增加一个额外的输入变量add_variable来辅助预测。'''
net = models.resnet50()
#其最后的输出层为：(fc): Linear(in_features=2048, out_features=1000, bias=True)
print(net)
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x = self.net(x)           #torch.Size([1, 1000])这就是resnet完全不动
        
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)  #torch.Size([1, 1001])
        #这里对外部输入变量"add_variable"进行unsqueeze操作是为了和net输出的tensor保持维度一致，
        # 常用于add_variable是单一数值 (scalar) 的情况，
        # 此时add_variable的维度是 (batch_size, )，
        # 需要在第二维补充维数1，从而可以和tensor进行torch.cat操作。    
        
        x = self.fc_add(x)   # torch.Size([1, 10])
        x = self.output(x)   #torch.Size([1, 10])
        return x
add_variable=torch.tensor([1,])
model=Model(net)    
x=torch.rand(1,3,224,224)
y=model(x,add_variable)   
y.shape    因为output中使用了softmax，所以现在y.sum()就是1


#5.3.3 添加输出
'''
除了模型最后的输出外，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。
基本的思路是修改模型定义中forward函数的return变量。也就是说return的时候把中间层的输出也给return出来
'''
'''
下面的例子实现的功能是：
在已经定义好的模型结构上，同时输出1000维的倒数第二层和10维的最后一层结果。
'''
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x1000 = self.net(x)                  #这是完全不变的resnet
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000     
net = models.resnet50()
model=Model(net)
x=torch.rand(1,3,224,224)
x10, x1000 =model(x)  #x10.shape= torch.Size([1, 10]),x1000.shape=torch.Size([1, 1000])

#5.4 模型保存和读取
'''
在很多场景下我们都会使用多GPU训练。
这种情况下，模型会分布于各个GPU上（参加2.3节分布数据式训练，这里暂不考虑分布模型式训练），
因为将数据拆分成子集放到不同的卡上，每个卡都是相同的model架构，避免之间通信

经过本节的学习，你将收获：
PyTorch的模型的存储格式
PyTorch如何存储模型
单卡与多卡训练下模型的保存与加载方法
'''
#5.4.1 存储格式
'''PyTorch存储模型主要采用pkl，pt，pth三种格式。就使用层面来说没有区别，
之前sklearn好像是pkl更多，pytorch中使用pt更多
'''


#5.4.2 模型存储内容
'''一个PyTorch模型主要包含两个部分：模型结构和权重。
其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）。
存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。'''

from torchvision import models
model = models.resnet152(pretrained=True)
# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重,其实更加推荐
torch.save(model.state_dict, save_dir)
'''对于PyTorch而言，pt, pth和pkl三种数据格式均支持模型权重和整个模型的存储，因此使用上没有差别。
save_dir是./model.pkl的格式   '''


#5.4.3 单卡和多卡模型存储的区别
'''PyTorch中将模型和数据放到GPU上有两种方式——.cuda()和.to(device)，
如果要使用多卡训练的话，需要对模型使用torch.nn.DataParallel。'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如果是多卡改成类似0,1,2
#目前对于没有显卡来说，还是使用.to(device)更好
model = model.cuda()  # 单卡model.to(device)
model = torch.nn.DataParallel(model).cuda()  # 多卡

#5.4.4 保存和加载的问题
#第一种:单卡保存+单卡加载,其实cpu版本也是这样的
'''在使用os.envision命令指定使用的GPU后，即可进行模型保存和读取操作。注意这里即便保存和读取时使用的GPU不同也无妨。'''
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model.cuda()

# 保存+读取整个模型
torch.save(model, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)   #加载训练好的模型参数
loaded_model = models.resnet152()   #实例化模型，但是这个模型参数是随机的
loaded_model.state_dict = loaded_dict  #直接对模型的参数进行赋值即可，此时的model就是预训练好的model了
loaded_model.cuda()


#第二种：单卡保存+多卡加载
'''这种情况的处理比较简单，读取单卡保存的模型后，使用nn.DataParallel函数进行分布式训练设置即可'''
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model.cuda()

# 保存+读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号
loaded_model = torch.load(save_dir)
loaded_model = nn.DataParallel(loaded_model).cuda()  #就是loaded_model.cuda()改成这一句

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_dict
loaded_model = nn.DataParallel(loaded_model).cuda()

#第三种：多卡保存+单卡加载
#第四种：多卡保存+多卡加载   这些暂时用不上




















