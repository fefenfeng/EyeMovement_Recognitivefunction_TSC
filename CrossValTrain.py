import torch
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch import nn
from ReadData import cross_val_load_data, MyDataset

# from Model.CNN1d import CNN1d
# from Model.CNN1d_GAP import CNN1d_GAP
# from Model.FCN import FCN
# from Model.ResNet import ResNet
from Model.InceptionTime import InceptionTime


# read data
data_all, labels_all, kfold = cross_val_load_data(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial1_sorted")

for fold, (train_ids, val_ids) in enumerate(kfold.split(data_all)):
    print("-------Fold {} begins!!!-------".format(fold + 1))
    # create sub dataset
    train_dataset = Subset(MyDataset(list(zip(data_all, labels_all))), train_ids)
    val_dataset = Subset(MyDataset(list(zip(data_all, labels_all))), val_ids)

    # dataset length
    train_dataset_len = len(train_dataset)
    print("The length of train dataset is :{}".format(train_dataset_len))
    val_dataset_len = len(val_dataset)
    print("The length of validation dataset is :{}".format(val_dataset_len))

    # build dataloader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # networks model initialisation

