# import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 1, 2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# test if net works
# if __name__ == '__main__':
#     # build instance
#     writer = SummaryWriter("../Logs_tensorboard/Models_Structure_Graph/FCN")
#     fcn = FCN()
#     # print(fcn)
#     #
#     #
#     # input = torch.ones((16, 2, 33920))  # 16的batch size，2通道,len33920
#     # print(input.shape)
#     # output = fcn(input)
#     # print(output.shape)
#     input = torch.randn((16, 2, 33920))  # 16的batch size，2通道,len33920
#
#     writer.add_graph(fcn, input)
#     writer.close()