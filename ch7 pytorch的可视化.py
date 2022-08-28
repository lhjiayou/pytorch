# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 21:50:44 2022

@author: 18721
"""

#7.1 可视化网络结构
'''另一个深度学习库Keras中可以调用一个叫做model.summary()的API来很方便地实现，
调用后就会显示我们的模型参数，输入大小，输出大小，模型的整体参数等，
但是在PyTorch中没有这样一种便利的工具帮助我们可视化我们的模型结构
但是开发了torchinfo工具包'''
#7.1.1 使用print函数打印模型基础信息
import torchvision.models as models
model = models.resnet18()  
#在resnet.py  文件中 ctrl+oad_state_dict_from_url,即可打开hub文件夹下的model_dir=r'D:/models/'
print(model)

# 7.1.2 使用torchinfo可视化网络结构
'''如何使用？
只需要使用torchinfo.summary()就行了，
必需的参数分别是model，input_size[batch_size,channel,h,w] 也就是输model及其输入层数据'''
from torchinfo import summary
resnet18 = models.resnet18() # 实例化模型
summary(resnet18, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽



#7.2 CNN的可视化
'''事实上CNN是最能够进行解释的'''
# 7.2.1 CNN卷积核可视化
'''卷积核在CNN中负责提取特征，可视化卷积核能够帮助人们理解CNN各个层在提取什么样的特征，
进而理解模型的工作原理。
例如在Zeiler和Fergus 2013年的paper中就研究了CNN各个层的卷积核的不同，
他们发现靠近输入的层提取的特征是相对简单的结构，
而靠近输出的层提取的特征就和图中的实体形状相近了
在keras的书中也是这么说的'''
'''在PyTorch中可视化卷积核也非常方便，
核心在于特定层的卷积核即特定层的模型权重，
可视化卷积核就等价于可视化对应的权重矩阵。'''
import torch
from torchvision.models import vgg11
'''由于from .._internally_replaced_utils import load_state_dict_from_url
中的load_state_dict_from_url(url, model_dir=r'D:/models/'，设置了保存的model的位置，
因此都会下载都非C盘的位置'''
model = vgg11(pretrained=True)
print(dict(model.features.named_children()))  #named_children()----返回的是子模块的迭代器

conv1 = dict(model.features.named_children())['3']
# Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
kernel_set = conv1.weight.detach()     #这个是没有grad的，从计算图上拆卸下来了


num = len(conv1.weight.detach())  
print(kernel_set.shape)  #torch.Size([128, 64, 3, 3])  卷积核数目64变成128，stride=3
#由于第“3”层的特征图由64维变为128维，因此共有128*64个卷积核

import matplotlib.pyplot as plt
for i in range(0,num):
    i_kernel = kernel_set[i]  #(64,3,3)
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:   #channel肯定大于1
        for idx, filer in enumerate(i_kernel):
            plt.subplot(8,8, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
            
#7.2.2 CNN特征图可视化方法
'''与卷积核相对应，输入的原始图像经过每次卷积层得到的数据称为特征图，
可视化卷积核是为了看模型提取哪些特征，      但是这个其实并不好说
可视化特征图则是为了看模型提取到的特征是什么样子的    结果说明一切'''

'''获取特征图的方法有很多种，可以从输入开始，逐层做前向传播，直到想要的特征图处将其返回。
尽管这种方法可行，但是有些麻烦了。
在PyTorch中，提供了一个专用的接口使得网络在前向传播过程中能够获取到特征图，这个接口的名称非常形象，叫做hook。

可以想象这样的场景，数据通过网络向前传播，网络某一层我们预先设置了一个钩子，
数据传播过后钩子上会留下数据在这一层的样子，读取钩子的信息就是这一层的特征图。'''
class Hook(object):
    '''首先实现了一个hook类
    就是用来保存layer的特征图'''
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []  #输入的特征数目
        self.features_out_hook = []  #输出的特征数目

    def __call__(self,module, fea_in, fea_out):  #初始化的时候需要输入特征和输出特征
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)  
        #这里的features_out_hook 是一个list，每次前向传播一次，都是调用一次，也就是features_out_hook 长度会增加1
        return None
    

def plot_feature(model, idx, inputs):
    hh = Hook()   #首先实例化
    model.features[idx].register_forward_hook(hh)
    
    # forward_model(model,False)
    model.eval()
    _ = model(inputs)
    print(hh.module_name)
    print((hh.features_in_hook[0][0].shape))
    print((hh.features_out_hook[0].shape))
    
    out1 = hh.features_out_hook[0]

    total_ft  = out1.shape[1]
    first_item = out1[0].cpu().clone()    

    plt.figure(figsize=(20, 17))
    

    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx+1) 
        
        plt.axis('off')
        #plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[ :, :].detach())
        
        
#7.2.3 CNN class activation map可视化方法
'''class activation map （CAM）的作用是判断哪些变量对模型来说是重要的，
在CNN可视化的场景下，即判断图像中哪些像素点对预测结果是重要的。
除了确定重要的像素点，人们也会对重要区域的梯度感兴趣，           重要的像素点
因此在CAM的基础上也进一步改进得到了Grad-CAM（以及诸多变种）。    重要的像素点的梯度

相比可视化卷积核与可视化特征图，CAM系列可视化更为直观，能够一目了然地确定重要区域，
进而进行可解释性分析或模型优化改进。CAM系列操作的实现可以通过开源工具包pytorch-grad-cam来实现
直接看热力图，显示哪儿的效果最好'''
# import torch
# from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# model = vgg11(pretrained=True)
# # from torchinfo import summary
# # summary(model, (1, 3, 224, 224))
# img_path = './dog.png'
# # resize操作是为了和传入神经网络训练图片大小一致
# img = Image.open(img_path).resize((224,224))
# # 需要将原始图片转为np.float32格式并且在0-1之间 
# rgb_img = np.float32(img)/255  #这个数据才能输入到model中去
# # img_tensor=torch.tensor(rgb_img)
# plt.imshow(img)

# # import pytorch_grad_cam   必须把__init__中的pytorch_grad_cam.metrics注释掉才不会报错
# from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# target_layers = [model.features[-1]]  #这是一个maxpool2d的特征图

# # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
# cam = GradCAM(model=model,target_layers=target_layers)
# targets = [ClassifierOutputTarget(200)]   
# # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
# grayscale_cam = cam(input_tensor=img_tensor, targets=targets)      这个img_tensor是什么？
# grayscale_cam = grayscale_cam[0, :]
# cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# print(type(cam_img))
# Image.fromarray(cam_img)


# 7.2.4 使用FlashTorch快速实现CNN可视化
'''可视化梯度'''
import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)  #模型
backprop = Backprop(model)

image = load_image('/content/images/great_grey_owl.jpg')  #图像
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)  #可视化

'''可视化卷积核'''
import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]  #Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv5_1_filters = [45, 271, 363, 489]
g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")  #比较慢



# 7.3 使用TensorBoard可视化训练过程
'''我们会结合训练集的损失函数和验证集的损失函数，绘制两条损失函数的曲线来确定训练的终点，
找到对应的模型用于测试。  这个就是要训练完成之后再来观察了
那么除了记录训练中每个epoch的loss值，能否实时观察损失函数曲线的变化，及时捕捉模型的变化呢'''

#7.3.3 TensorBoard的配置与启动
# 在使用TensorBoard前，我们需要先指定一个文件夹供TensorBoard保存记录下来的数据。
# 然后调用tensorboard中的SummaryWriter作为上述“记录员”
from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs')

# tensorboard --logdir=./runs --host=8080
