import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self,x):
        shortcut = self.bn_shortcut(self.shortcut(x))

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = self.relu3(x + shortcut)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = ResNetBlock(2, 64)
        self.block2 = ResNetBlock(64, 64)
        self.block3 = ResNetBlock(64, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# # test if net works
# if __name__ == '__main__':
#     # build instance
#     writer = SummaryWriter("../Logs_tensorboard/Models_Structure_Graph/Resnet")
#     resnet = ResNet()
#     # print(resnet)
#     #
#     # input = torch.ones((16, 2, 33920))  # 16的batch size，2通道,len33920
#     # print(input.shape)
#     # output = resnet(input)
#     # print(output.shape)
#
#     input = torch.randn((16, 2, 33920))  # 16的batch size，2通道,len33920
#
#     writer.add_graph(resnet, input)
#     writer.close()
