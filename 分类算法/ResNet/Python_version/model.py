import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, AdaptiveAvgPool2d

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, if_downsample=False):
        super(Bottleneck, self).__init__()
        if if_downsample:
            self.down_sample = nn.Sequential(
                Conv2d(in_channel, out_channel*4, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel*4)
            )
        else:
            self.down_sample = nn.Sequential()
        self.residual = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            ReLU(inplace=True),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            ReLU(inplace=True),
            Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        identity = self.down_sample(x)
        residual += identity
        out = self.relu(residual)
        return out


class ResNet101(nn.Module):
    def __init__(self, num_layers, NUM_CLASS):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1)
        )
        self.conv2_3 = nn.Sequential()
        self.conv3_4 = nn.Sequential()
        self.conv4_23 = nn.Sequential()
        self.conv5_3 = nn.Sequential()

        self.make_layer(self.conv2_3, 64, 64, num_layers[0], 1)
        self.make_layer(self.conv3_4, 256, 128, num_layers[1], 2)
        self.make_layer(self.conv4_23, 512, 256, num_layers[2], 2)
        self.make_layer(self.conv5_3, 1024, 512, num_layers[3], 2)

        self.end_layer = nn.Sequential(
            AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, NUM_CLASS)
        )

    def make_layer(self, block, in_channel, out_channel, stack_num, stride):
        block.add_module('layer_{}'.format(0), Bottleneck(in_channel, out_channel, stride, True))
        for i in range(1, stack_num + 1):
            block.add_module('layer_{}'.format(i), Bottleneck(out_channel * 4, out_channel, 1, False))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_3(x)
        x = self.conv3_4(x)
        x = self.conv4_23(x)
        x = self.conv5_3(x)   # 7*7*2048
        x = self.end_layer(x)
        return x

# if __name__ == "__main__":
#     resnet = ResNet101([3, 4, 23, 3], 10)
#     a = torch.Tensor(1, 3, 224, 224)
#     output = resnet(a)
#     print(output.size())