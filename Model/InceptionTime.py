import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter


# no bottleneck transfer original data
def pass_though(x):
    return x


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_size=32):
        super(Inception, self).__init__()
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels=in_channels,
                                        out_channels=bottleneck_size,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False
                                        )
        else:
            self.bottleneck = pass_though
            bottleneck_size = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(in_channels=bottleneck_size,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[0],
                                                stride=1,
                                                padding=kernel_sizes[0]//2,
                                                bias=False
                                                )
        self.conv_from_bottleneck_2 = nn.Conv1d(in_channels=bottleneck_size,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[1],
                                                stride=1,
                                                padding=kernel_sizes[1]//2,
                                                bias=False
                                                )
        self.conv_from_bottleneck_3 = nn.Conv1d(in_channels=bottleneck_size,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[2],
                                                stride=1,
                                                padding=kernel_sizes[2]//2,
                                                bias=False
                                                )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_maxpool = nn.Conv1d(in_channels=in_channels,
                                           out_channels=n_filters,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False)
        self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        x_maxpool = self.max_pool(x)

        x1 = self.conv_from_bottleneck_1(x_bottleneck)
        x2 = self.conv_from_bottleneck_2(x_bottleneck)
        x3 = self.conv_from_bottleneck_3(x_bottleneck)
        x4 = self.conv_from_maxpool(x_maxpool)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.activation(self.batch_norm(x))
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_size=32, use_residual=True):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.inception_1 = Inception(in_channels=in_channels,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_size=bottleneck_size,
                                     )
        self.inception_2 = Inception(in_channels=4*n_filters,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_size=bottleneck_size,
                                     )
        self.inception_3 = Inception(in_channels=4*n_filters,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_size=bottleneck_size,
                                     )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=4*n_filters,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                nn.BatchNorm1d(
                    num_features=4*n_filters
                )
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.inception_1(x)
        z = self.inception_2(z)
        z = self.inception_3(z)
        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)
        return z


class InceptionTime(nn.Module):
    def __init__(self):
        super(InceptionTime, self).__init__()
        self.model = nn.Sequential(
            InceptionBlock(
                in_channels=2,
                n_filters=4,
                kernel_sizes=[9, 19, 39],
                bottleneck_size=32,
                use_residual=True
            ),
            InceptionBlock(
                in_channels=4*4,
                n_filters=4,
                kernel_sizes=[9, 19, 39],
                bottleneck_size=32,
                use_residual=True
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(4*4*1, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# test if net works
# if __name__ == '__main__':
#     # build instance
#     writer = SummaryWriter("../Logs_tensorboard/Models_Structure_Graph/InceptionTime")
#     inceptiontime = InceptionTime()
#     # print(inceptiontime)
#
#     input = torch.randn((16, 2, 33920))  # 16的batch size，2通道,len33920
#     # print(input.shape)
#     # output = inceptiontime(input)
#     # print(output.shape)
#
#     writer.add_graph(inceptiontime, input)
#     writer.close()