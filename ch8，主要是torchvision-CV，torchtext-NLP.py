# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:24:15 2022

@author: 18721
"""

#8.1 简介
'''于计算机视觉，有TorchVision、TorchVideo等用于图片和视频处理；
对于自然语言处理，有torchtext；
对于图卷积网络，有PyTorch Geometric ······'''

#8.2 torchvision
# 8.2.1 torchvision简介
'''我们经常会用到torchvision来调用预训练模型，加载数据集，对图片进行数据增强的操作。
torchvision包含了在计算机视觉中常常用到的数据集，模型和图像处理的方式，
而具体的torchvision则包括了下面这几部分：

torchvision.datasets *   数据集

torchvision.models *  预训练模型

torchvision.tramsforms *   图像的变换方式

torchvision.io

torchvision.ops

torchvision.utils

'''

# 8.2.2 torchvision.datasets
'''torchvision.datasets主要包含了一些我们在计算机视觉中常见的数据集'''

# 8.2.3 torchvision.transforms
'''则需要进行归一化和大小缩放等操作，这些是常用的数据预处理方法。
除此之外，当图片数据有限时，我们还需要通过对现有图片数据进行各种变换，
如缩小或放大、水平或垂直翻转等，这些是常见的数据增强方法。'''
#之前哪一章说到实际竞赛中，有另外一个数据增强的包
from torchvision import transforms
data_transform = transforms.Compose([
    transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# 8.2.4 torchvision.models
# https://github.com/pytorch/vision/tree/main/torchvision/models
'''模型包括Classification
Semantic Segmentation
Object Detection，instance Segmentation and Keypoint Detection
Video classification
'''


# 8.2.5 torchvision.io
'''在torchvision.io提供了视频、图片和文件的 IO 操作的功能，它们包括读取、写入、编解码处理操作。'''


# 8.2.6 torchvision.ops
# 8.2.7 torchvision.utils




#8.3 PyTorchVideo简介
'''有关视频的深度学习模型仍然有着许多缺点：

计算资源耗费更多，并且没有高质量的model zoo，不能像图片一样进行迁移学习和论文复现。
数据集处理较麻烦，但没有一个很好的视频处理工具。
随着多模态越来越流行，亟需一个工具来处理其他模态'''
#8.3.1 PyTorchVideo的主要部件和亮点
#8.3.2 PyTorchVideo的安装
'''pip install pytorchvideo
安装的虚拟环境的python版本 >= 3.7
PyTorch >= 1.8.0，安装的torchvision也需要匹配
CUDA >= 10.2
ioPath：具体情况
fvcore版本 >= 0.1.4：具体情况
'''
# 8.3.3 Model zoo 和 benchmark
# 8.3.4 使用 PyTorchVideo model zoo
'''
PyTorchVideo提供了三种使用方法，并且给每一种都配备了tutorial
TorchHub，这些模型都已经在TorchHub存在。我们可以根据实际情况来选择需不需要使用预训练模型。
除此之外，官方也给出了TorchHub使用的 tutorial 。

PySlowFast，使用 PySlowFast workflow 去训练或测试PyTorchVideo models/datasets.

PyTorch Lightning建立一个工作流进行处理，点击查看官方 tutorial。
'''

# 8.4 torchtext简介
'''PyTorch官方用于自然语言处理（NLP）的工具包torchtext'''
# 8.4.1 torchtext的主要组成部分
'''torchtext可以方便的对文本进行预处理，例如截断补长、构建词表等。
torchtext主要包含了以下的主要组成部分：

数据处理工具 torchtext.data.functional、torchtext.data.utils
数据集 torchtext.data.datasets
词表工具 torchtext.vocab
评测指标 torchtext.metrics'''

# 8.4.2 torchtext的安装
'''pip install torchtext'''

# 8.4.3 构建数据集
#8.4.3.1 Field及其使用
'''
Field是torchtext中定义数据类型以及转换为张量的指令。
torchtext 认为一个样本是由多个字段（文本字段，标签字段）组成，
不同的字段可能会有不同的处理方式，所以才会有 Field 抽象。
定义Field对象是为了明确如何处理不同类型的数据，但具体的处理则是在Dataset中完成的。'''
from torchtext import data
tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)
'''​ sequential设置数据是否是顺序表示的；

​ tokenize用于设置将字符串标记为顺序实例的函数

​ lower设置是否将字符串全部转为小写；

​ fix_length设置此字段所有实例都将填充到一个固定的长度，方便后续处理；

​ use_vocab设置是否引入Vocab object，如果为False，
则需要保证之后输入field中的data都是numerical的'''
#接下来构建dataset
import tqdm  #就是进度条
def get_dataset(csv_data, text_field, label_field, test=False):
    fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", text_field), ("toxic", label_field)]       
    examples = []

    if test:
        # 如果为测试集，则不加载label
        for text in tqdm(csv_data['comment_text']):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields
'''这里使用数据csv_data中有"comment_text"和"toxic"两列，分别对应text和label。'''
import pandas as pd
train_data = pd.read_csv('train_toxic_comments.csv')
valid_data = pd.read_csv('valid_toxic_comments.csv')
test_data = pd.read_csv("test_toxic_comments.csv")
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)
# 得到构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
valid_examples, valid_fields = get_dataset(valid_data, TEXT, LABEL)
test_examples, test_fields = get_dataset(test_data, TEXT, None, test=True)
# 构建Dataset数据集
train = data.Dataset(train_examples, train_fields)
valid = data.Dataset(valid_examples, valid_fields)
test = data.Dataset(test_examples, test_fields)

'''基本流程为：
定义Field对象完成后，通过get_dataset函数可以读入数据的文本和标签，
将二者（examples）连同field一起送到torchtext.data.Dataset类中，
即可完成数据集的构建。使用以下命令可以看下读入的数据情况：'''
# 检查keys是否正确
print(train[0].__dict__.keys())
print(test[0].__dict__.keys())
# 抽查内容是否正确
print(train[0].comment_text)


#8.4.3.2 词汇表（vocab）
'''在NLP中，将字符串形式的词语（word）转变为数字形式的向量表示（embedding）是非常重要的一步，
被称为Word Embedding。词嵌入

这一步的基本思想是收集一个比较大的语料库（尽量与所做的任务相关），
在语料库中使用word2vec之类的方法构建词语到向量（或数字）的映射关系，
之后将这一映射关系应用于当前的任务，将句子中的词语转为向量表示。
但是bert中好像是自己学习这个映射关系了
'''
# 在torchtext中可以使用Field自带的build_vocab函数完成词汇表构建。
TEXT.build_vocab(train)

#8.4.3.3 数据迭代器
'''其实就是torchtext中的DataLoader
torchtext支持只对一个dataset和同时对多个dataset构建数据迭代器。'''
from torchtext.data import Iterator, BucketIterator
# 若只针对训练集构造迭代器
# train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)

# 同时对训练集和验证集进行迭代器的构建
train_iter, val_iter = BucketIterator.splits(
        (train, valid), # 构建数据集所需的数据集
        batch_sizes=(8, 8),
        device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False
)

test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False)

#8.4.3.4 使用自带数据集
'''与torchvision类似，torchtext也提供若干常用的数据集方便快速进行算法测试。'''


# 8.4.4 评测指标（metric）
'''NLP中部分任务的评测不是通过准确率等指标完成的，
比如机器翻译任务常用BLEU (bilingual evaluation understudy) score
来评价预测文本和标签文本之间的相似程度。
torchtext中可以直接调用torchtext.data.metrics.bleu_score来快速实现BLEU，
就是比较特殊的metrics'''
from torchtext.data.metrics import bleu_score
candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
bleu_score(candidate_corpus, references_corpus)


# 8.4.5 其他
'''对于文本研究而言，当下Transformer已经成为了绝对的主流，
因此PyTorch生态中的HuggingFace等工具包也受到了越来越广泛的关注。'''