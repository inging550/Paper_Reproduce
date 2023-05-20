import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, NUM_CLASS, init_weight):   # NUM_CLASS的值根具实际情况更改，init_weight为一个布尔值（代表是否初始化权重参数，一般为True）
        super(VGG16, self).__init__()
        # 定义网络结构
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASS)
        )
        if init_weight:
            self.define_weight()

    # 前向传播函数
    def forward(self, x):
        x = self.features(x)  # (3,224,224) -> (256,6,6)
        x = torch.flatten(x, start_dim=1)  # (256,6,6) -> (9216)  # 9216 = 256*6*6
        x = self.classifier(x)  # (9216) -> (10)
        return x

    def define_weight(self):  # 权重初始化参数
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out')
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)
            elif isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, 0, 0.01)  # 全连接层初始为正态分布均值为0方差为0.01
                nn.init.constant_(i.bias, 0)  # 用单位矩阵来填充2维输入张量或变量。在线性层尽可能多的保存输入特性。
