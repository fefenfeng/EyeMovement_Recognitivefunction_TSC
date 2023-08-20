import os
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from itertools import groupby
# from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset
# from torch.utils.tensorboard import SummaryWriter


def load_single_file_with_index(file_path, label, index_range):
    df = pd.read_csv(file_path).iloc[index_range[0]:index_range[1]]
    data_values = df.values
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_values)
    data = [(standardized_data, label)]
    return data


def load_single_file_as_dataset(file_path, label):
    df = pd.read_csv(file_path)
    data_values = df.values
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_values)
    data = [(standardized_data, label)]
    return data


def load_data_with_index(path, index_range):
    dataset = []
    scaler = StandardScaler()
    for label in ['0', '1']:
        dir_path = os.path.join(path, label)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path).iloc[index_range[0]:index_range[1]]
            standardized_data = scaler.fit_transform(df.values)
            dataset.append((standardized_data, label))

    return dataset


def load_all_data(path):
    dataset = []
    scaler = StandardScaler()
    for label in ['0', '1']:
        dir_path = os.path.join(path, label)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path)
            standardized_data = scaler.fit_transform(df.values)
            dataset.append((standardized_data, label))

    return dataset

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

    # create a list of participants sorted by data points count
    sorted_participants = sorted(participants_data.keys(), key=lambda x: len(participants_data[x]), reverse=True)
    # Randomize participants with the same data points count
    randomized_participants = []
    for _, group in groupby(sorted_participants, key=lambda x: len(participants_data[x])):
        group_list = list(group)
        random.shuffle(group_list)
        randomized_participants.extend(group_list)
    # Split by participant into train val test
    train_size = int(0.8 * sum([len(participants_data[pid]) for pid in randomized_participants]))
    train_set, val_set = [], []
    current_train_size = 0
    train_ids, val_ids = [], []

    for pid in randomized_participants:
        if current_train_size + len(participants_data[pid]) <= train_size:
            train_set.extend(participants_data[pid])
            current_train_size += len(participants_data[pid])
            train_ids.append(pid)
        else:
            val_set.extend(participants_data[pid])
            val_ids.append(pid)
    return train_set, val_set, train_ids, val_ids


def modified_cross_val_load_data(path):
    # Load data
    participants_data = {}

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

    # Create a list of participants
    participant_ids = list(participants_data.keys())

    # Create a "label" for stratification based on participants' data points count
    stratify_labels = [len(participants_data[pid]) for pid in participant_ids]

    # Create a new StratifiedKFold object for the user to iterate over
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return participants_data, participant_ids, stratify_labels, skf


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        features, label = self.data[index]
        return torch.tensor(features.T, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)
        # feature + label --> tensor，float32,long,transpose（len，2）--> (2，len）

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#
#     alldata = load_all_data(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial0_sorted")
#     test_dataset = MyDataset(alldata)
#     batch_size = 16
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



#     testdata = load_data_with_index(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial0\Prosaccades_gap", [0, 6400])
#     test_dataset = MyDataset(testdata)
#     batch_size = 16
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # data001_1 = load_single_file_as_dataset(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial1_sorted\0\001_1.csv", '0')
    #
    # train_dataset = MyDataset(data001_1)
    # # build dataloader
    # batch_size = 16
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     # test data instance in dataloader
#     for sample in train_loader:
#         positions, targets = sample
#         print(positions.shape)
#         print(targets)
    # train_dataset_len = len(train_dataset)
    # print("The length of train dataset is :{}".format(train_dataset_len))


#     # read data
#     participants_data, participant_ids, stratify_labels, stratified_kfold = modified_cross_val_load_data(
#         r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial1_sorted")
#     for fold, (train_pids_idx, val_pids_idx) in enumerate(stratified_kfold.split(participant_ids, stratify_labels)):
#         print("-------Fold {} begins!!!-------".format(fold + 1))
#
#         # Get the actual data for these participant IDs
#         train_data = [data for pid_idx in train_pids_idx for data in participants_data[participant_ids[pid_idx]]]
#         val_data = [data for pid_idx in val_pids_idx for data in participants_data[participant_ids[pid_idx]]]
#
#         # # Print the participant IDs for train and validation set
#         # print("Train Participant IDs:", [participant_ids[pid_idx] for pid_idx in train_pids_idx])
#         # print("Validation Participant IDs:", [participant_ids[pid_idx] for pid_idx in val_pids_idx])
#
#         # Create sub dataset
#         train_dataset = MyDataset(train_data)
#         val_dataset = MyDataset(val_data)

        # # dataset length
        # train_dataset_len = len(train_dataset)
        # print("The length of train dataset is :{}".format(train_dataset_len))
        # val_dataset_len = len(val_dataset)
        # print("The length of validation dataset is :{}".format(val_dataset_len))

        # # build dataloader
        # batch_size = 16
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



    # # load data
    # train_data, val_data, train_ids, val_ids = modified_load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
    #                                                         r"\data_info\trial1_sorted")
    # print("Number of participants in dataset are :{}".format(len(train_ids)))
    # print("Number of participants in dataset are :{}".format(len(val_ids)))
    # print(train_ids)
    # print(val_ids)

    # # dataset instantiation
    # train_dataset = MyDataset(train_data)
    # val_dataset = MyDataset(val_data)
    # test_dataset = MyDataset(test_data)

    # # build dataloader
    # batch_size = 16
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # # test dataset.__len__
    # train_dataset_len = len(train_dataset)
    # print("length of train dataset is :{}".format(train_dataset_len))
    # val_dataset_len = len(val_dataset)
    # print("length of validation dataset is :{}".format(val_dataset_len))
    # test_dataset_len = len(test_dataset)
    # print("length of train dataset is :{}".format(test_dataset_len))

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
