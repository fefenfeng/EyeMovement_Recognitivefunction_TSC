# import torch
from torch import nn
# from ReadData import *
# from torch.utils.tensorboard import SummaryWriter


class CNN1d_GAP(nn.Module):
    def __init__(self):
        super(CNN1d_GAP, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            # nn.Dropout(p=0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.model(x)
        # for layer in self.model:
        #     x = layer(x)
        #     print(x.shape)
        return x


# test if net works
# if __name__ == '__main__':
#     writer = SummaryWriter("../Logs_tensorboard/Models_Structure_Graph/CNN1d_GAP")
#     cnn1d_gap = CNN1d_GAP()
#     input = torch.randn((16, 2, 33920))  # 16的batch size，2通道,len33920
#
#     writer.add_graph(cnn1d_gap, input)
#     writer.close()

    # # build instance
    # cnn1d = CNN1d()
#     print(cnn1d)
#     # build dataset and dataloader
#     train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
#                                                             r"\data_info\trial1_sorted")
#     # dataset instantiation
#     train_dataset = MyDataset(train_data)
#     val_dataset = MyDataset(val_data)
#     test_dataset = MyDataset(test_data)
#     # build dataloader
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#     for batch in test_loader:
#         positions, targets = batch
#         print(f'Test batch - features shape: {positions.shape}, labels shape: {targets.shape}')
#         input0 = positions
#         print(input0.shape)
#         output0 = cnn1d(input0)
#         print(output0.shape)
#
#
    # input = torch.ones((16, 2, 33920))  # 16的batch size，2通道,len33920
    # print(input.shape)
    # output = cnn1d(input)
    # print(output.shape)
