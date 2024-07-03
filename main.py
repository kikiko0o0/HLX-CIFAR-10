#!/usr/bin/env python
# coding: utf-8
#hlx 2024

# In[8]:


import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

plt.ion()  # 打开交互模式，使得绘图实时显示

# 设置数据集的下载链接
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

demo = False  # 如果为True，则使用小型数据集

# 设置数据集路径
if demo:
    data_dir = d2l.download_extract('cifar10_tiny') 
else:
    data_dir = 'D:\\cifar-10\\cifar10\\data\\cifar-10' 

def read_csv_labels(fname):
    """读取CSV文件中的标签信息"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]  # 跳过第一行表头
    tokens = [l.rstrip().split(',') for l in lines]  # 分割每行的标签信息
    return dict(((name, label) for name, label in tokens))  # 返回字典形式的标签

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))  # 读取训练集标签
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)  # 创建目标目录
    shutil.copy(filename, target_dir)  # 复制文件

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    n = collections.Counter(labels.values()).most_common()[-1][1]  # 获取最少类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))  # 每个类别分配的验证集样本数
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):  # 遍历训练集文件
        label = labels[train_file.split('.')[0]]  # 获取文件对应的标签
        fname = os.path.join(data_dir, 'train', train_file)  # 获取文件路径
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))  # 复制到train_valid目录
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))  # 复制到验证集目录
            label_count[label] = label_count.get(label, 0) + 1  # 更新标签计数
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))  # 复制到训练集目录
    return n_valid_per_label  # 返回每个类别分配的验证集样本数

def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):  # 遍历测试集文件
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))  # 复制到测试集目录

def reorg_cifar10_data(data_dir, valid_ratio):
    """重组 CIFAR-10 数据集"""
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))  # 读取训练集标签
    reorg_train_valid(data_dir, labels, valid_ratio)  # 重新组织训练集和验证集
    reorg_test(data_dir)  # 重新组织测试集

batch_size = 32 if demo else 128  # 设置批量大小
valid_ratio = 0.1  # 验证集比例
reorg_cifar10_data(data_dir, valid_ratio)  # 重组数据集

# 定义训练数据的增强和预处理
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),  # 调整图像大小为40x40
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪到32x32
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),  # 转换为张量
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 归一化
])

# 定义测试数据的预处理
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转换为张量
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 归一化
])

# 创建数据集和数据加载器
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train) for folder in ['train', 'train_valid']]  # 训练集和训练验证集

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test) for folder in ['valid', 'test']]  # 验证集和测试集

# 创建数据加载器
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]  # 训练集和训练验证集的数据加载器

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)  # 验证集的数据加载器
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)  # 测试集的数据加载器

# 定义SE模块
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1)  # 第一个全连接层
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1)  # 第二个全连接层

    def forward(self, x):
        out = F.avg_pool2d(x, x.size(2))  # 全局平均池化
        out = F.relu(self.fc1(out))  # ReLU激活
        out = torch.sigmoid(self.fc2(out))  # Sigmoid激活
        return x * out  # 通道加权

# 定义ResNet-50模型中的瓶颈块
class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)  # 1x1卷积
        self.bn1 = nn.BatchNorm2d(planes)  # 批量归一化
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 3x3卷积
        self.bn2 = nn.BatchNorm2d(planes)  # 批量归一化
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)  # 批量归一化

        self.se = SEBlock(self.expansion * planes, reduction)  # SE模块

        self.shortcut = nn.Sequential()  # 快捷连接
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),  # 1x1卷积
                nn.BatchNorm2d(self.expansion * planes)  # 批量归一化
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 1x1卷积 + ReLU激活
        out = F.relu(self.bn2(self.conv2(out)))  # 3x3卷积 + ReLU激活
        out = self.bn3(self.conv3(out))  # 1x1卷积
        out = self.se(out)  # SE模块
        out += self.shortcut(x)  # 加上快捷连接
        out = F.relu(out)  # ReLU激活
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 初始卷积层
        self.bn1 = nn.BatchNorm2d(64)  # 批量归一化
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 第一层
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 第二层
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 第三层
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 第四层
        self.linear = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 设置步幅
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 添加瓶颈块
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 初始卷积 + ReLU激活
        out = self.layer1(out)  # 第一层
        out = self.layer2(out)  # 第二层
        out = self.layer3(out)  # 第三层
        out = self.layer4(out)  # 第四层
        out = F.avg_pool2d(out, 4)  # 平均池化
        out = out.view(out.size(0), -1)  # 展平
        out = self.linear(out)  # 全连接层
        return out
    
def ResNet50(num_classes=10):
    """定义ResNet-50模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def get_net():
    """获取ResNet-50模型"""
    num_classes = 10
    net = ResNet50(num_classes)
    return net

loss = nn.CrossEntropyLoss(reduction="none")  # 定义损失函数

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)  # 学习率调度器
    num_batches, timer = len(train_iter), d2l.Timer()  # 获取批次数和计时器
    legend = ['train loss', 'train acc']  # 设置图例
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)  # 动画显示训练过程
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # 多GPU训练
    for epoch in range(num_epochs):
        net.train()  # 训练模式
        metric = d2l.Accumulator(3)  # 累加器
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)  # 训练一个批次
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
                animator.show()
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)  # 验证准确性
            animator.add(epoch + 1, (None, None, valid_acc))
            animator.show()
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
    plt.show()

def save_model(net, file_path='D:\\cifar-10\\cifar10\\model.pth'):
    """保存模型"""
    torch.save(net.state_dict(), file_path)

print("Is CUDA available:", torch.cuda.is_available())  # 检查CUDA是否可用  

devices, num_epochs, lr, wd = d2l.try_all_gpus(), 100, 0.1e-4, 5e-4  # 设置设备、训练周期、学习率、权重衰减
lr_period, lr_decay, net = 50, 0.1, get_net()  # 设置学习率调度周期和衰减率，获取模型
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)  # 训练模型



# In[9]:


net, preds = get_net(), []  # 获取模型，初始化预测列表
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)  # 使用训练验证集重新训练模型
save_model(net, 'D:\\cifar-10\\cifar10\\model.pth')  # 保存模型

for X, _ in test_iter:  # 对测试集进行预测
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())  # 获取预测结果并存入列表
sorted_ids = list(range(1, len(test_ds) + 1))  # 创建测试集索引
sorted_ids.sort(key=lambda x: str(x))  # 对索引进行排序
df = pd.DataFrame({'id': sorted_ids, 'label': preds})  # 创建数据框
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])  # 映射预测标签
df.to_csv('submission.csv', index=False)  
