import os
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch


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
    train_data, val_data, test_data = [], [], []
    for label in data:
        train, temp = train_test_split(data[label], test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_data.extend([(x, label) for x in train])  # 将train中的data带上标签组成元组存入train_data
        val_data.extend([(x, label) for x in val])
        test_data.extend([(x, label) for x in test])

    return train_data, val_data, test_data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        features, label = self.data[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)
        # 转换feature和label到tensor，float32和long

    def __len__(self):
        return len(self.data)


def create_dataloaders(path, batch_size=32):
    train_data, val_data, test_data = load_and_process_data(path)

    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)
    test_dataset = MyDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial0_sorted", batch_size=32)

    # # 查看tensor尺寸
    # for batch in train_loader:
    #     features, labels = batch
    #     print(f'Train batch - features shape: {features.shape}, labels shape: {labels.shape}')
    #
    # for batch in val_loader:
    #     features, labels = batch
    #     print(f'Validation batch - features shape: {features.shape}, labels shape: {labels.shape}')
    #
    # for batch in test_loader:
    #     features, labels = batch
    #     print(f'Test batch - features shape: {features.shape}, labels shape: {labels.shape}')

    # 查看tensor数据
    # for i, (features, labels) in enumerate(train_loader):
    #     for j in range(len(features)):
    #         print(f'Feature {j}: {features[j]}')
    #         a = features[j]
    #         print(f'Label {j}: {labels[j]}')
    #         b = labels[j]
    #         print('---')
    #     # 只打印第一个批次的数据
    #     if i == 0:
    #         break
