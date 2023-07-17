import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter


def load_and_process_data(path):
    # Load data
    data = {}       # 空字典存储加载数据
    for label in ['0', '1']:
        data[label] = []
        dir_path = os.path.join(path, label)    # 拼接正负样本文件夹完整路径
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)   # 拼接完整文件路径
            df = pd.read_csv(file_path)     # csv-->dataframe
            data[label].append(df.values)   # dataframe值，理应为np.array二维

    # Standardize data
    scaler = StandardScaler()
    for label in data:
        for i in range(len(data[label])):
            data[label][i] = scaler.fit_transform(data[label][i])   # 对每个类别每个数据进行标准化

    # Split into train, val, test
    train_set, val_set, test_set = [], [], []
    for label in data:
        train, temp = train_test_split(data[label], test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_set.extend([(x, label) for x in train])  # 将train中的data带上标签组成元组存入train_data
        val_set.extend([(x, label) for x in val])
        test_set.extend([(x, label) for x in test])

    return train_set, val_set, test_set


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        features, label = self.data[index]
        return torch.tensor(features.T, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)
        # 转换feature和label到tensor，float32和long,.T转置将数据从（len，2），转为（2，len）

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#     # load data
#     train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
#                                                             r"\data_info\trial1_sorted")
#     # 实例化dataset
#     train_dataset = MyDataset(train_data)
#     val_dataset = MyDataset(val_data)
#     test_dataset = MyDataset(test_data)
#     # 创建dataloader
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # # 测试dataset len函数
    # train_dataset_len = len(train_dataset)
    # print("训练数据集长度为:{}".format(train_dataset_len))
    # val_dataset_len = len(val_dataset)
    # print("验证数据集长度为:{}".format(val_dataset_len))
    # test_dataset_len = len(test_dataset)
    # print("测试数据集长度为:{}".format(test_dataset_len))
    # # 测试dataset中数据
    # position, target = train_dataset[0]
    # print(position.shape)
    # print(target)
    # # 检验是否标准化
    # position, target = train_dataset[0]
    # mean = torch.mean(position, dim=1)
    # std = torch.std(position, dim=1)
    # print("均值: ", mean)
    # print("标准差: ", std)
    # # 测试loader中的数据
    # for sample in train_loader:
    #     positions, targets = sample
    #     print(positions.shape)
    #     print(targets)

    # # 查看loader中tensor尺寸
    # for batch in train_loader:
    #     positions, targets = batch
    #     print(f'Train batch - features shape: {positions.shape}, labels shape: {targets.shape}')
    #
    # for batch in val_loader:
    #     positions, targets = batch
    #     print(f'Validation batch - features shape: {positions.shape}, labels shape: {targets.shape}')
    #
    # for batch in test_loader:
    #     positions, targets = batch
    #     print(f'Test batch - features shape: {positions.shape}, labels shape: {targets.shape}')

    # # 查看tensor数据
    # for i, (positions, targets) in enumerate(train_loader):
    #     for j in range(len(positions)):
    #         print(f'Feature {j}: {positions[j]}')
    #         a = positions[j]
    #         print(f'Label {j}: {targets[j]}')
    #         b = targets[j]
    #         print('---')
    #     # 只打印第一个批次的数据
    #     if i == 0:
    #         break
