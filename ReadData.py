import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
# from torch.utils.data import Subset
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter


def load_and_process_data(path):
    # Load data
    data = {}       # new dic to load data
    for label in ['0', '1']:
        data[label] = []
        dir_path = os.path.join(path, label)    # sample folders full path
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)   # full file path
            df = pd.read_csv(file_path)     # csv-->dataframe
            data[label].append(df.values)   # dataframe value, which is supposed to be np.array two-dimensional

    # Standardize data
    scaler = StandardScaler()
    for label in data:
        for i in range(len(data[label])):
            data[label][i] = scaler.fit_transform(data[label][i])   # Standardise each data for each class

    # Split into train, val, test
    train_set, val_set, test_set = [], [], []
    for label in data:
        train, temp = train_test_split(data[label], test_size=0.3, random_state=10)
        val, test = train_test_split(temp, test_size=0.5, random_state=10)
        train_set.extend([(x, label) for x in train])  # Stores tuples of data from train with labels into train_data
        val_set.extend([(x, label) for x in val])
        test_set.extend([(x, label) for x in test])

    return train_set, val_set, test_set


def cross_val_load_data(path):
    # Load data
    data = {}       # new dic to load data
    for label in ['0', '1']:
        data[label] = []
        dir_path = os.path.join(path, label)    # sample folders full path
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)   # full file path
            df = pd.read_csv(file_path)     # csv-->dataframe
            data[label].append(df.values)   # dataframe value, which is supposed to be np.array two-dimensional

    # Standardize data
    scaler = StandardScaler()
    for label in data:
        for i in range(len(data[label])):
            data[label][i] = scaler.fit_transform(data[label][i])   # Standardise each data for each class

    data_all = data['0'] + data['1']
    labels_all = ['0'] * len(data['0']) + ['1'] * len(data['1'])

    # k-fold split
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return data_all, labels_all, stratified_kfold


def modified_load_and_process_data(path):
    # Load data
    participants_data = {}  # Dict to group data by participant

    for label in ['0', '1']:
        dir_path = os.path.join(path, label)
        for file_name in os.listdir(dir_path):
            participant_id = file_name.split('_')[0]
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path)

            # Standardize data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(df.values)

            # Add to the participant's data
            if participant_id not in participants_data:
                participants_data[participant_id] = []
            participants_data[participant_id].append((standardized_data, label))

    # Split by participant into train, val, test
    train_set, val_set, test_set = [], [], []

    participant_ids = list(participants_data.keys())
    train_ids, temp_ids = train_test_split(participant_ids, test_size=0.3, random_state=10)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=10)

    for pid in train_ids:
        train_set.extend(participants_data[pid])
    for pid in val_ids:
        val_set.extend(participants_data[pid])
    for pid in test_ids:
        test_set.extend(participants_data[pid])

    return train_set, val_set, test_set


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        features, label = self.data[index]
        return torch.tensor(features.T, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)
        # feature + label --> tensor，float32,long,transpose（len，2）--> (2，len）

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    # load data
    train_data, val_data, test_data = modified_load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
                                                            r"\data_info\trial1_sorted")
    # dataset instantiation
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)
    test_dataset = MyDataset(test_data)

    # build dataloader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # test dataset.__len__
    train_dataset_len = len(train_dataset)
    print("length of train dataset is :{}".format(train_dataset_len))
    val_dataset_len = len(val_dataset)
    print("length of validation dataset is :{}".format(val_dataset_len))
    test_dataset_len = len(test_dataset)
    print("length of train dataset is :{}".format(test_dataset_len))

    # # data instance shape in dataset test
    # position, target = train_dataset[0]
    # print(position)
    # print(position.shape)
    # print(target)
    # print(target.shape)

    # # test if data has been standardisation
    # position, target = train_dataset[0]
    # mean = torch.mean(position, dim=1)
    # std = torch.std(position, dim=1)
    # print("Mean: ", mean)
    # print("Standard deviation: ", std)

    # # test data instance in dataloader
    # for sample in val_loader:
    #     positions, targets = sample
    #     print(positions.shape)
    #     print(targets)

    # # size of tensor in dataloader
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

    # # check full tensor data
    # for i, (positions, targets) in enumerate(train_loader):
    #     for j in range(len(positions)):
    #         print(f'Feature {j}: {positions[j]}')
    #         a = positions[j]
    #         print(f'Label {j}: {targets[j]}')
    #         b = targets[j]
    #         print('---')
    #     # only print first batch data
    #     if i == 0:
    #         break
