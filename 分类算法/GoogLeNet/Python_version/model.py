import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, AdaptiveAvgPool2d
import torch.nn.functional as F


class GoogleNet(nn.Module):
    def __init__(self, NUM_CLASS, init_weight):
        super(GoogleNet, self).__init__()
        self.block1 = nn.Sequential(
            Conv2d(3, 64, 7, 2, 3),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),

            Conv2d(64, 64, 1, 1),
            Conv2d(64, 192, 3, 1, 1),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1, ceil_mode=False)
        )
        self.block2 = nn.Sequential(
            inception([192, 64, 96, 128, 16, 32, 32]),
            inception([256, 128, 128, 192, 32, 96, 64]),
            MaxPool2d(3, 2),
            inception([480, 192, 96, 208, 16, 48, 64])
        )
        self.block3 = nn.Sequential(
            # 输入做一个辅助分类器
            inception([512, 160, 112, 224, 24, 64, 64]),

            inception([512, 128, 128, 256, 24, 64, 64]),
            inception([512, 112, 144, 288, 32, 64, 64])
        )
        self.block4 = nn.Sequential(
            # 对输入做一个辅助分类器
            inception([528, 256, 160, 320, 32, 128, 128]),

            MaxPool2d(3, 2, 1, ceil_mode=False),
            inception([832, 256, 160, 320, 32, 128, 128]),
            inception([832, 384, 192, 384, 48, 128, 128]),
            AdaptiveAvgPool2d((1, 1))
        )
        self.end_block = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, NUM_CLASS)
        )
        if init_weight:
            self.define_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.end_block(x)
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


class inception(nn.Module):
    def __init__(self, input):
        super(inception, self).__init__()
        self.path1 = Conv2d(input[0], input[1], 1, 1)
        self.path2 = nn.Sequential(
            Conv2d(input[0], input[2], 1, 1),
            ReLU(inplace=True),
            Conv2d(input[2], input[3], 3, 1, 1),
            ReLU(inplace=True),
        )
        self.path3 = nn.Sequential(
            Conv2d(input[0], input[4], 1, 1),
            ReLU(inplace=True),
            Conv2d(input[4], input[5], 5, 1, 2),
            ReLU(inplace=True)
        )
        self.path4 = nn.Sequential(
            MaxPool2d(3, 1, 1),
            Conv2d(input[0], input[6], 1, 1)
        )

    def forward(self, x):
        x1 = F.relu(self.path1(x))
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = F.relu(self.path4(x))
        return torch.cat((x1, x2, x3, x4), dim=1)
