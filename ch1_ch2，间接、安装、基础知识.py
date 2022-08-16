# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:24:53 2022

@author: 18721
"""

import torch
import numpy as np


#2.1节
#2.1.1 简介
# 在PyTorch中， torch.Tensor 是存储和变换数据的主要工具。如果你之前用过NumPy，
# 你会发现 Tensor 和NumPy的多维数组非常类似。然而，Tensor 提供GPU计算和自动求梯度等更多功能，
# 这些使 Tensor 这一数据类型更加适合深度学习



#2.1.2 创建tensor
#随机初始化矩阵 我们可以通过torch.rand()的方法，构造一个随机初始化的矩阵：
x = torch.rand(4, 3)   #就像np.random.rand一样使用，目前是没有grad的
print(x)
# 通过torch.zeros()构造一个矩阵全为 0，并且通过dtype设置数据类型为 long。
# 除此以外，我们还可以通过torch.zero_()和torch.zeros_like()将现有矩阵转换为全0矩阵.
x = torch.zeros(4, 3, dtype=torch.long)
print(x)
print(x.size())  #torch.Size([4, 3])
print(x.shape)   #torch.Size([4, 3])
print(x.size(0))  #输出的就是4
print(x.shape[0])   #输出的就是4
'''参考链接：https://blog.csdn.net/blueblood7/article/details/125971528
shape 是属性，size() 是方法。
shape 是 numpy 中的 array 和 pytorch 中的 tensor 通用的。size() 只能用在 tensor 上。
shape[i]、size(i) 和 size()[i] 结果是一样的。'''

#事实上用torch.from_numpy也可以直接获得tensor
x=np.arange(12).reshape(-1,3)
x=torch.from_numpy(x)
print(x)


#2.1.3 张量的操作
#1.索引操作：(类似于numpy)，利用copy方法创建副本避免相互干扰问题的出现
# 需要注意的是：索引出来的结果与原数据共享内存，修改一个，另一个会跟着修改。如果不想修改，可以考虑使用copy()等方法

#2.维度变换 张量的维度变换常见的方法有torch.view()和torch.reshape()，但是reshape确实用习惯了
#torch.view() 返回的新tensor与源tensor共享内存(其实是同一个tensor)，
# 更改其中的一个，另外一个也会跟着改变。(顾名思义，view()仅仅是改变了对这个张量的观察角度)
#所以说需要注意避免干扰出现

#！！！！多情况下，我们希望原始张量和变换后的张量互相不影响。
# 推荐的方法是我们先用 clone() 创造一个张量副本然后再使用 torch.view()进行函数维度变换 。

#有一个元素 tensor ，我们可以使用 .item() 来获得这个 value，而不获得其他性质
#对于多个元素的tensor，如果我们想看数值，可以只用.detach().numpy()



#2.1.4 广播机制，这个是numpy最好用的技能了
x = torch.arange(1, 3).view(1, 2)
print(x)  #一行两列，shape=(1,2),会被广播为(3,2),也就是行复制3次
y = torch.arange(1, 4).view(3, 1)
print(y)  #三行一列,shape=(3,1)，会被广播成(3,2),也就是列复制2次
print(x + y)




#2.2节
# 所有神经网络的核心是 autograd 包
#autograd简介

# 如果设置它的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。
# 当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。
# 这个张量的所有梯度将会自动累加到.grad属性。
#事实上计算图中requires_grad默认就是true

#但也不都是需要梯度的
# 要阻止一个张量被跟踪历史，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。
# 为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad(): 中。
# 在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，
# 但是我们不需要在此过程中对他们进行梯度计算。
'''问：with torch.no_grad():和net.eval()的差异：
参考链接：https://blog.csdn.net/weixin_43593330/article/details/108486214

进行validation时，会使用model.eval()切换到测试模式，在该模式下，
主要用于通知dropout层和batchnorm层在train和val模式间切换。
在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); 
batchnorm层会继续计算数据的mean和var等参数并更新。
在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，
直接使用在训练阶段已经学出的mean和var值。
该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行backprobagation。

而with torch.no_grad()则主要是用于停止autograd模块的工作，
以起到加速和节省显存的作用，具体行为就是停止gradient计算，
从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。

那么是不是如果想要使用MC dropout，就是将验证代码放到with torch.no_grad()中
'''
#2.2.1 梯度
# out 是一个标量，因此out.backward()和 out.backward(torch.tensor(1.)) 等价。这个其实不重要
#因为NN的训练loss，正常就是标量，那么就是简单的loss.backward()即可，一般是不会出现非标量的loss的

#注意梯度累加问题
# 注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，
# 梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
#不过！！！有的时候为了节约计算成本，我在知乎上看过有人选择不清零，以用比较小的计算资源实现比较好的计算性能


#如果我们想要修改 tensor 的数值，但是又不希望被 autograd 记录(即不会影响反向传播)， 
# 那么我们可以对 tensor.data 进行操作。
#不过，这个要求不是很常见
x = torch.ones(1,requires_grad=True) #x是tensor([1.], requires_grad=True)
print(x.data) # 还是一个tensor，输出的是tensor([1.])
print(x.data.requires_grad) # 但是已经是独立于计算图之外，因此现在是false
y = 2 * x   #y是有梯度的 tensor([2.], grad_fn=<MulBackward0>)
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()   #计算梯度是不会报错的
print(x) # 更改data的值也会影响tensor的值   tensor([100.], requires_grad=True)，
print(x.grad) #仍然是tensor([2.])，并没有因为更改了x.data的数值而影响这个值

#2.3并行计算
#不过李沐刚说的装机视频说到，多张卡因为之间带宽受限，事实上1+1<2
#编写程序中，当我们使用了 .cuda() 时，其功能是让我们的模型或者数据从CPU迁移到GPU(0)当中，通过GPU开始计算。
#因为都是cpu跑的，所以不太记得，好像是什么to_device()让data和model都迁移到GPU上？

#设置默认显卡
#设置在文件最开始部分
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "2" # 设置默认的显卡

#设置多块显卡
CUDA_VISBLE_DEVICE=0,1 python train.py # 使用0，1两块GPU

#现在的并行计算方式是：
# 不再拆分模型，我训练的时候模型都是一整个模型。但是我将输入的数据拆分。
# 所谓的拆分数据就是，同一个模型在不同GPU中训练一部分数据，然后再分别计算一部分数据之后，
# 只需要将输出的数据做一个汇总，然后再反传。
#这种方式可以解决之前模式遇到的通讯问题。现在的主流方式是数据并行的方式(Data parallelism)
 