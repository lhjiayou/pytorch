# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:52:35 2022

@author: 18721
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#6.1 自定义损失函数
#6.1.1 函数的形式
def my_loss(output, target):
    '''这其实即使mse_loss'''
    loss = torch.mean((output - target)**2)
    return loss

x=torch.rand(10,1)
y=torch.rand(10,1)


mse=nn.MSELoss()
loss1=mse(x,y)
# loss1.item() 0.2542099356651306

loss2=my_loss(x,y)
# loss2.item() 0.2542099356651306
loss1.item()==loss2.item()


#6.1.2 类的形式
# 我们可以将其当作神经网络的一层来对待，同样地，我们的损失函数类就需要继承自nn.Module类
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
    def forward(self,inputs,targets,smooth=1):
        '''这个smooth的作用其实就是为了避免除以0'''
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# 使用方法    
criterion = DiceLoss()   #就是实例化这个损失函数
input=torch.rand(10,1,requires_grad=True)
targets=torch.rand(10,1,requires_grad=True)
loss = criterion(input,targets)
'''
UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
  就像F.dropout也是被nn.dropout给取代了
'''


# 除此之外，常见的损失函数还有BCE-Dice Loss，Jaccard/Intersection over Union (IoU) Loss，Focal Loss......

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                     
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
# --------------------------------------------------------------------
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
# --------------------------------------------------------------------
    
ALPHA = 0.8
GAMMA = 2
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
# 更多的可以参考链接1  https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/notebook
#事实上我们只要能够提供损失函数的表达式，自然能够实现函数形式或者类的形式

'''
对于定义损失函数的时候，是使用numpy还是torch的建议：
在自定义损失函数时，涉及到数学运算时，我们最好全程使用PyTorch提供的张量计算接口，
这样就不需要我们实现自动求导功能并且我们可以直接调用cuda，使
用numpy或者scipy的数学运算时，操作会有些麻烦
这个时候其实是不在计算图中的，那么怎么根据loss实现BP呢
'''


#6.2 动态调整学习率
'''可以通过一个适当的学习率衰减策略来改善这种现象，提高我们的精度。这种设置方式在PyTorch中被称为scheduler'''
#6.2.1 使用官方scheduler
'''PyTorch已经在torch.optim.lr_scheduler为我们封装好了一些动态调整学习率的方法供我们使用
lr_scheduler.LambdaLR

lr_scheduler.MultiplicativeLR

lr_scheduler.StepLR

lr_scheduler.MultiStepLR

lr_scheduler.ExponentialLR

lr_scheduler.CosineAnnealingLR

lr_scheduler.ReduceLROnPlateau

lr_scheduler.CyclicLR

lr_scheduler.OneCycleLR

lr_scheduler.CosineAnnealingWarmRestarts
！！！！
需要注意的是：
注：
我们在使用官方给出的torch.optim.lr_scheduler时，需要将scheduler.step()放在optimizer.step()后面进行使用。
'''
#基本流程为：
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
	scheduler1.step() 
	...
    schedulern.step()
    
#6.2.2 自定义scheduler
'''自定义函数adjust_learning_rate来改变param_group中lr的值
假设我们现在正在做实验，需要学习率每30轮下降为原来的1/10
'''
from torchvision import models
model = models.resnet18(pretrained=True)
#因为修改了hub目录下的def load_state_dict_from_url(url, model_dir=r'D:/models/')
# Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to D:/models/resnet50-0676ba61.pth

optimizer = torch.optim.Adam(model.parameters()) 
'''现在optimizer的params_group是optimizer.param_groups，6个参数组成的字典
param_group['lr'] = lr是其中的一对键值对
'''
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
    
    
#6.3 模型微调，用的是torchvision
'''迁移学习：
迁移学习的一大应用场景是模型微调（finetune）。
简单来说，就是我们先找到一个同类的别人训练好的模型，把别人现成的训练好了的模型拿过来，
换成自己的数据，通过训练调整一下参数。 
在PyTorch中提供了许多预训练好的网络模型（VGG，ResNet系列，mobilenet系列......），
这些模型都是PyTorch官方在相应的大型数据集训练好的。
学习如何进行模型微调，可以方便我们快速使用预训练模型完成自己的任务。'''

#6.3.1 模型微调的流程
'''在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即源模型。这一步是直接由pytorch提供的

创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。
我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。
我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。输出层作为顶层的架构，是task_specific的

为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。

在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。

其实这个部分应该包括两种方式，一种是预训练模型的底层架构只作为特征提取器，这一种很简单
但是底层模型是完全不能动的，另一种是以极小的学习率来调整底层的特征提取器的
'''

#6.3.2 下载已经预训练的model
'''传递pretrained参数
通过True或者False来决定是否使用预训练好的权重，在默认状态下pretrained = False，
意味着我们不使用预训练得到的权重，当pretrained = True，意味着我们将使用在一些数据集上预训练得到的权重。'''
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)

'''下载预训练模型的一些注意事项：
通常PyTorch模型的扩展为.pt或.pth，程序运行时会首先检查默认路径中是否有已经下载的模型权重，
一旦权重被下载，下次加载就不需要下载了。

一般情况下预训练模型的下载会比较慢，我们可以直接通过迅雷或者其他方式去 这里 查看自己的模型里面model_urls，
然后手动下载，预训练模型的权重在Linux和Mac的默认下载路径是用户根目录下的.cache文件夹。
在Windows下就是C:\Users\<username>\.cache\torch\hub\checkpoint。  我现在已经修改了
我们可以通过使用 torch.utils.model_zoo.load_url()设置权重的下载地址。

如果觉得麻烦，还可以将自己的权重下载下来放到同文件夹下，然后再将参数加载网络。

self.model = models.resnet50(pretrained=False)  
self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
这其实就是模型加载的常见方式
'''

#6.3.3 训练特定层
'''
在默认情况下，参数的属性.requires_grad = True，如果我们从头开始训练或微调不需要注意这里。
但如果我们正在提取特征并且只想为新初始化的层计算梯度，其他参数不进行改变。
那我们就需要通过设置requires_grad = False来冻结部分层。
也就是我们冻结预训练模型的所有底层，将输出层改成我们任务所需要的输出层
然后对这个层进行训练
'''
def set_parameter_requires_grad(model, feature_extracting):
    '''这个函数就是对model的权重进行冻结的'''
    if feature_extracting:  
        #feature_extracting即特征提取，也就是我们将model的权重参数冻结，不可训练，仅作为特征提取器
        for param in model.parameters():
            param.requires_grad = False  #冻结
            
'''使用resnet18为例的将1000类改为4类，但是仅改变最后一层的模型参数，不改变特征提取的模型参数；
注意我们先冻结模型参数的梯度，再对模型输出部分的全连接层进行修改，这样修改后的全连接层的参数就是可计算梯度的。
因为默认就是可以修改的'''

import torchvision.models as models
# 冻结参数的梯度
feature_extract = True  #就是说只能作为特征提取器
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad  #现在是true
set_parameter_requires_grad(model, feature_extract)   
param.requires_grad   #现在是false
# 修改模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=num_ftrs, out_features=4, bias=True)  #修改了model的fc
# 之后在训练过程中，model仍会进行梯度回传，但是参数更新则只会发生在fc层。通过设定参数的requires_grad属性，我们完成了指定训练模型的特定层的目标，这对实现模型微调非常重要。
for param in model.parameters():
    print(param.requires_grad)

#6.3 模型微调，用的是timm
'''除了使用torchvision.models进行预训练以外，还有一个常见的预训练模型库，叫做timm，
这个库是由来自加拿大温哥华Ross Wightman创建的。
里面提供了许多计算机视觉的SOTA模型，可以当作是torchvision的扩充版本，并且里面的模型在准确度上也较高。'''
import timm
model_list=timm.list_models()  #长度612
#1.查看已经有的预训练模型
'''# 查看特定模型的所有种类 每一种系列可能对应着不同方案的模型，比如Resnet系列就包括了ResNet18，50，101等模型，
# 我们可以在timm.list_models()传入想查询的模型名称（模糊查询），比如我们想查询densenet系列的所有模型。'''
all_densnet_models = timm.list_models("*densenet*")   #模糊查询
all_densnet_models

#2.查看模型的具体参数 当我们想查看下模型的具体参数的时候，我们可以通过访问模型的default_cfg属性来进行查看
model = timm.create_model('resnet34',num_classes=10,pretrained=True)  #这个下载的时候，没有tqdm，而且很慢，不如直接用github下载到本地更好
model.default_cfg
'''
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
 'num_classes': 1000,
 'input_size': (3, 224, 224),  输入大小
 'pool_size': (7, 7),  池化窗口大小
 'crop_pct': 0.875,
 'interpolation': 'bilinear',
 'mean': (0.485, 0.456, 0.406),   这可能是对图像所做的处理
 'std': (0.229, 0.224, 0.225),
 'first_conv': 'conv1',
 'classifier': 'fc',
 'architecture': 'resnet34'}
'''
model = timm.create_model('resnet34',pretrained=True)  #这是已经预训练好的模型
x = torch.randn(1,3,224,224)
output = model(x)  #那么这个输出有没有经过softmax呢
output.shape  #torch.Size([1, 1000])，因为就是1000类
model.layer4[0].conv1 

list(dict(model.named_children())['conv1'].parameters())[0].shape  #torch.Size([64, 3, 7, 7])
list(model.conv1.parameters())[0].shape  #torch.Size([64, 3, 7, 7])

model = timm.create_model('resnet34',num_classes=10,pretrained=True)  #直接在新建model的时候将num_class=10
#此时最后的fc (fc): Linear(in_features=512, out_features=10, bias=True)
x = torch.randn(1,3,224,224)
output = model(x)
output.shape


# 改变输入通道数（比如我们传入的图片是单通道的，但是模型需要的是三通道图片） 我们可以通过添加in_chans=1来改变
model = timm.create_model('resnet34',num_classes=10,pretrained=True,in_chans=1)
model.default_cfg
x = torch.randn(1,1,100,100)  #其实图片的长宽也不是必须为224
output = model(x)
output.shape  #torch.Size([1, 10])

#模型的保存
# timm库所创建的模型是torch.model的子类，我们可以直接使用torch库中内置的模型参数保存和加载的方法
torch.save(model.state_dict(),'./checkpoint/timm_model.pth')
model.load_state_dict(torch.load('./checkpoint/timm_model.pth'))

#6.4 半精度训练
'''
GPU的性能主要分为两部分：算力和显存，前者决定了显卡计算的速度，
后者则决定了显卡可以同时放入多少数据用于计算。

PyTorch默认的浮点数存储方式用的是torch.float32，
小数点后位数更多固然能保证数据的精确性，
但绝大多数场景其实并不需要这么精确，只保留一半的信息也不会影响结果，也就是使用torch.float16格式。
由于数位减了一半，因此被称为“半精度”。
有两个问题：

如何在PyTorch中设置半精度训练

使用半精度训练的注意事项
'''
#其一，怎么设置半精度训练
'''在PyTorch中使用autocast配置半精度训练，同时需要在下面三处加以设置：

第一，import autocast
from torch.cuda.amp import autocast

第二，模型设置
在模型定义中，使用python的装饰器方法，用autocast装饰模型中的forward函数。关于装饰器的使用，可以参考这里：
@autocast()   
def forward(self, x):
    ...
    return x

第三，训练过程
在训练过程中，只需在将数据输入模型及其之后的部分放入“with autocast():“即可：
 for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)'''
        
#其二，注意：
'''半精度训练主要适用于数据本身的size比较大（比如说3D图像、视频等）。
当数据本身的size并不大时（比如手写数字MNIST数据集的图片尺寸只有28*28），
使用半精度训练则可能不会带来显著的提升。'''

# 6.5 数据增强-imgaug
'''
其实还是CV部分，数据增强比较好做
 在计算视觉领域，生成增强图像相对容易。即使引入噪声或裁剪图像的一部分，模型仍可以对图像进行分类，
 数据增强有一系列简单有效的方法可供选择，有一些机器学习库来进行计算视觉领域的数据增强，
 比如：imgaug 官网它封装了很多数据增强算法，给开发者提供了方便。
'''

#6.5.1 imgaug简介
#6.5.1.1 imgaug简介
'''imgaug是计算机视觉任务中常用的一个数据增强的包，相比于torchvision.transforms，它提供了更多的数据增强方法，因此在各种竞赛中，人们广泛使用imgaug来对数据进行增强操作。imgaug是计算机视觉任务中常用的一个数据增强的包，相比于torchvision.transforms，
它提供了更多的数据增强方法，因此在各种竞赛中，人们广泛使用imgaug来对数据进行增强操作。'''

#6.5.1.2conda安装
conda config --add channels conda-forge
conda install imgaug


#6.5.2 imgaug的使用
'''imgaug仅仅提供了图像增强的一些方法，但是并未提供图像的IO操作，
因此我们需要使用一些库来对图像进行导入，建议使用imageio进行读入，

如果使用的是opencv进行文件读取的时候，需要进行手动改变通道，将读取的BGR图像转换为RGB图像。
用PIL.Image进行读取时，因为读取的图片没有shape的属性，所以需要将读取到的img转换为np.array()的形式再进行处理'''
#其一，单张图片的处理，其实不常用，更多的还是需要使用批量
import imageio  #实现图片的读入，读出
import imgaug as ia
# %matplotlib inline #内嵌画图，因为使用spyder，所以并不需要

# 图片的读取
img = imageio.imread("./Lenna.jpg")   #(316,316,3),那这个不是channe在最后么
# 使用Image进行读取
# img = Image.open("./Lenna.jpg")
# image = np.array(img)
# ia.imshow(image)



# mgaug包含了许多从Augmenter继承的数据增强的操作
#单一数据增强
from imgaug import augmenters as iaa
# 设置随机数种子
ia.seed(4)
# 实例化方法
rotate = iaa.Affine(rotate=(-4,45))
img_aug = rotate(image=img)
ia.imshow(img_aug)
        

#混合的数据增强操作，和compose很像
# 对一张图片做多种数据增强处理。这种情况下，我们就需要利用imgaug.augmenters.Sequential()
# 来构造我们数据增强的pipline，该方法与torchvison.transforms.Compose()相类似
iaa.Sequential(children=None, # Augmenter集合
               random_order=False, # 是否对每个batch使用不同顺序的Augmenter list
               name=None,
               deterministic=False,
               random_state=None)
# 构建处理序列
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25,25)),
    iaa.AdditiveGaussianNoise(scale=(10,60)),
    iaa.Crop(percent=(0,0.2))
])
# 对图片进行处理，image不可以省略，也不能写成images
image_aug = aug_seq(image=img)
ia.imshow(image_aug)



#批量数据增强，对批次图片进行处理
# 在实际使用中，我们通常需要处理更多份的图像数据。
# 此时，可以将图形数据按照NHWC的形式或者由列表组成的HWC的形式对批量的图像进行处理。
# 主要分为以下两部分，对批次的图片以同一种方式处理和对批次的图片进行分部分处理。
'''对一个批次的图片，相同的增强方式'''
#其一，一种增强处理
import numpy as np
images = [img,img,img,img,]
images_aug = rotate(images=images)
ia.imshow(np.hstack(images_aug))
#其二，多种增强处理，类似于transforms.compose,不过可见此时其实每张图片处理之后是不同的
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Crop(percent=(0, 0.2))
])
# 传入时需要指明是images参数
images_aug = aug_seq.augment_images(images = images)
#images_aug = aug_seq(images = images) 
ia.imshow(np.hstack(images_aug))

'''对批次的图片分部分处理
imgaug相较于其他的数据增强的库，有一个很有意思的特性，
即就是我们可以通过imgaug.augmenters.Sometimes()对batch中的一部分图片应用一部分Augmenters,
剩下的图片应用另外的Augmenters。'''

iaa.Sometimes(p=0.5,  # 代表划分比例
              then_list=None,  # Augmenter集合。p概率的图片进行变换的Augmenters。
              else_list=None,  #1-p概率的图片会被进行变换的Augmenters。注意变换的图片应用的Augmenter只能是then_list或者else_list中的一个。
              name=None,
              deterministic=False,
              random_state=None)

#不同大小的图片
# 构建pipline
seq = iaa.Sequential([
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
    iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    iaa.Cutout()  # replace one squared area within the image by a constant intensity value
], random_order=True)

# 加载不同大小的图片
images_different_sizes = [
    imageio.imread("./1.jpg"),
    imageio.imread("./2.jpg"),
    imageio.imread("./3.jpg"),
]
# [i.shape for i in images_different_sizes]  可见是完全不同大小的三张图片
# [(257, 286, 3), (536, 800, 3), (289, 520, 3)]

# 对图片进行增强
images_aug = seq(images=images_different_sizes)

# 可视化结果
print("Image 0 (input shape: %s, output shape: %s)" % (images_different_sizes[0].shape, images_aug[0].shape))
ia.imshow(np.hstack([images_different_sizes[0], images_aug[0]]))

print("Image 1 (input shape: %s, output shape: %s)" % (images_different_sizes[1].shape, images_aug[1].shape))
ia.imshow(np.hstack([images_different_sizes[1], images_aug[1]]))

print("Image 2 (input shape: %s, output shape: %s)" % (images_different_sizes[2].shape, images_aug[2].shape))
ia.imshow(np.hstack([images_different_sizes[2], images_aug[2]]))


#6.5.3 PyTorch中如何使用imgaug的模板
import numpy as np
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 构建pipline
tfs = transforms.Compose([
    iaa.Sequential([
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.MultiplyBrightness(mul=(0.65, 1.35)),
    ]).augment_image,
   
    transforms.ToTensor()   # 不要忘记了使用ToTensor()
])

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, n_images, n_classes, transform=None): #实例化的时候transform=tfs
		# 图片的读取，建议使用imageio，下面是使用的随机数
        self.images = np.random.randint(0, 255,
                                        (n_images, 224, 224, 3),
                                        dtype=np.uint8)
        self.targets = np.random.randn(n_images, n_classes)
        self.transform = transform

    def __getitem__(self, item):
        image = self.images[item]
        target = self.targets[item]

        if self.transform:
            image = self.transform(image)  #变换

        return image, target

    def __len__(self):
        return len(self.images)


def worker_init_fn(worker_id):  
    '''关于num_workers在Windows系统上只能设置成0，但是当我们使用Linux远程服务器时，
    可能使用不同的num_workers的数量，这是我们就需要注意worker_init_fn()函数的作用了。
    它保证了我们使用的数据增强在num_workers>0时是对数据的增强是随机的。'''
    ia.seed(np.random.get_state()[1][0] + worker_id)
worker_init_fn(1)

custom_ds = CustomDataset(n_images=50, n_classes=10, transform=tfs)
custom_dl = DataLoader(custom_ds, batch_size=64,
                       num_workers=0, pin_memory=True, )
                       # worker_init_fn=worker_init_fn)


#6.5.4 总结
# 数据扩充是我们需要掌握的基本技能，除了imgaug以外，我们还可以去学习其他的数据增强库，
# 包括但不局限于Albumentations，Augmentor。


# 6.6 使用argparse进行调参，目前不是很需要，选择性不看
'''如何更方便的修改超参数是我们需要考虑的一个问题。
这时候，要是有一个库或者函数可以解析我们输入的命令行参数再传入模型的超参数中该多好。
到底有没有这样的一种方法呢？答案是肯定的，这个就是 Python 标准库的一部分：Argparse。

argparse的简介

argparse的使用

如何使用argparse修改超参数'''
#6.6.1 argparse简介
# argsparse是python的命令行解析的标准模块，内置于python，不需要安装。
# 这个库可以让我们直接在命令行中就可以向程序中传入参数。
# 我们可以使用python file.py来运行python文件。         在命令行中运行python文件
# 而argparse的作用就是将命令行传入的其他参数进行解析、保存和使用。
# 在使用argparse后，我
# 们在命令行输入的参数就可以以这种形式python file.py --lr 1e-4 --batch_size 32  修改超参数
# 来完成对常见超参数的设置。








