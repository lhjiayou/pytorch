# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:49:58 2022

@author: 18721
"""

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
# %matplotlib inline  内嵌画图的，因为使用spyder所以不需要



###############1.加载原始图片
img = Image.open("lenna.jpg") 
print(img.size)  #(316, 316)，这个读取的图片其实没有显示channel，本来应该是C*H*W
plt.imshow(img)



###############2.中心裁剪transforms.CenterCrop(size)
# 对给定图片进行沿中心切割
# 对图片沿中心放大切割，超出图片大小的部分填0，会填充黑色
img_centercrop1 = transforms.CenterCrop((500,500))(img)
print(img_centercrop1.size)  #(500, 500)

# 对图片沿中心缩小切割，超出期望大小的部分剔除
img_centercrop2 = transforms.CenterCrop((224,224))(img)
print(img_centercrop2.size)  #(224, 224)

plt.subplot(1,3,1),plt.imshow(img),plt.title("Original")
plt.subplot(1,3,2),plt.imshow(img_centercrop1),plt.title("500 * 500")
plt.subplot(1,3,3),plt.imshow(img_centercrop2),plt.title("224 * 224")
plt.show()



###############3.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# 对图片的亮度，对比度，饱和度，色调进行改变
img_CJ = transforms.ColorJitter(brightness=1,contrast=0.5,saturation=0.5,hue=0.5)(img)
print(img_CJ.size)   #(316, 316)
plt.imshow(img_CJ)



#############4.transforms.Grayscale(num_output_channels)  让图片变成channel=1或者channel=3
img_grey_c3 = transforms.Grayscale(num_output_channels=3)(img)
img_grey_c1 = transforms.Grayscale(num_output_channels=1)(img)
#他们都是(316,316)的图像，但是channel不同
# transforms.ToTensor()(img_grey_c3).size()  torch.Size([3, 316, 316])
# transforms.ToTensor()(img_grey_c1).size()   torch.Size([1, 316, 316])

plt.subplot(1,2,1),plt.imshow(img_grey_c3),plt.title("channels=3")
plt.subplot(1,2,2),plt.imshow(img_grey_c1),plt.title("channels=1")
plt.show()



###########5.transforms.Resize  等比缩放
# 等比缩放
img_resize = transforms.Resize(224)(img)
print(img_resize.size)  #(224, 224)
plt.imshow(img_resize)



########6.transforms.Scale
# 等比缩放 不推荐使用此转换以支持调整大小
img_scale = transforms.Scale(224)(img)
print(img_scale.size)
plt.imshow(img_scale)



########7.transforms.RandomCrop  随机裁剪，这个是数据增强的重要手段
# 随机裁剪成指定大小
# 设立随机种子
import torch
torch.manual_seed(31)
# 随机裁剪
img_randowm_crop1 = transforms.RandomCrop(224)(img)
img_randowm_crop2 = transforms.RandomCrop(224)(img)
# transforms.ToTensor()(img_randowm_crop1)==transforms.ToTensor()(img_randowm_crop2)
print(img_randowm_crop1.size) #(224, 224)
plt.subplot(1,2,1),plt.imshow(img_randowm_crop1)
plt.subplot(1,2,2),plt.imshow(img_randowm_crop2)
plt.show()
#两张图片并不相等，但是因为设置了随机数种子，所以每次的图片都相同


#############8.transforms.RandomHorizontalFlip   随机水平翻转
# 随机左右旋转
# 设立随机种子，可能不旋转
import torch
torch.manual_seed(31)
img_random_H = transforms.RandomHorizontalFlip()(img)
print(img_random_H.size)  #(316, 316)
plt.imshow(img_random_H)



###########9.transforms.RandomVerticalFlip   随机垂直翻转
# 随机垂直方向旋转
img_random_V = transforms.RandomVerticalFlip()(img)
print(img_random_V.size)  #(316, 316)
plt.imshow(img_random_V)



############10.transforms.RandomResizedCrop  随机裁剪成指定的大小
# 随机裁剪成指定大小，其实每次都是不一样的
img_random_resizecrop = transforms.RandomResizedCrop(224,scale=(0.5,0.5))(img)
print(img_random_resizecrop.size)
plt.imshow(img_random_resizecrop)

############对图片进行组合变化 tranforms.Compose()
# 对一张图片的操作可能是多种的，我们使用transforms.Compose()将他们组装起来
transformer = transforms.Compose([
    transforms.Resize(256),   #调整大小
    transforms.transforms.RandomResizedCrop((224), scale = (0.5,1.0)),  #随机裁剪
    transforms.RandomVerticalFlip(),  #随机垂直翻转
])
img_transform = transformer(img)
plt.imshow(img_transform)










