import torch
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch import nn
from ReadData import cross_val_load_data, MyDataset
from numpy import std

# from Model.CNN1d import CNN1d
# from Model.CNN1d_GAP import CNN1d_GAP
from Model.FCN import FCN
# from Model.ResNet import ResNet
# from Model.InceptionTime import InceptionTime


# read data
data_all, labels_all, stratified_kfold = cross_val_load_data(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial1_sorted")

# to store best val acc and loss per fold
val_best_loss_5fold = []
val_best_acc_5fold = []

for fold, (train_ids, val_ids) in enumerate(stratified_kfold.split(data_all, labels_all)):
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
    fcn = FCN()
    if torch.cuda.is_available():
        fcn = fcn.cuda()  # 转移到cuda上

    # define loss function
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # define optimizer
    learning_rate = 1e-3
    # l2penalty = 1e-3
    # optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate, weight_decay=l2penalty)  # add L2 regularization
    optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)

    # learning rate reduce
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                           min_lr=0.0001)
    # Set up for some parameters in training
    total_train_step = 0  # 训练次数
    total_val_step = 0  # 测试次数
    epoch = 1500  # 训练轮数

    # early stopping
    best_val_loss = float('inf')
    best_val_acc = float('inf')
    patience_counter = 0
    patience_limit = 50

    # tensorboard
    writer = SummaryWriter(f"./Logs_tensorboard/FCN_5folds_1st/Fold_{fold + 1}")
    start_time = time.time()

    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        # training begin
        fcn.train()  # turn to training mode
        total_train_loss = 0
        total_train_accuracy = 0
        for data in train_loader:
            positions, targets = data  # feature+label-->data
            if torch.cuda.is_available():
                positions = positions.cuda()
                targets = targets.cuda()

            outputs = fcn(positions)
            loss = loss_fn(outputs, targets)  # calculate loss
            total_train_loss = total_train_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_train_accuracy = total_train_accuracy + accuracy

            optimizer.zero_grad()  # turn optimizer gradient--> zero
            loss.backward()  # backward propagation
            optimizer.step()  # Step gradient update

            total_train_step = total_train_step + 1

            # train loss step
            if total_train_step % 50 == 0:
                end_time = time.time()
                train_time = end_time - start_time
                print(
                    "Total train step: {}, Train time: {}, Loss: {}".format(total_train_step, train_time, loss.item()))
                writer.add_scalar("Train_loss_step", loss.item(), total_train_step)

        scheduler.step(total_train_loss)  # lr reduce

        # total train loss and acc
        print("Total train Loss: {}".format(total_train_loss))
        print("Total train accuracy: {}".format(total_train_accuracy / train_dataset_len))
        writer.add_scalar("Train_loss", total_train_loss, total_val_step)
        writer.add_scalar("Train_accuracy", total_train_accuracy / train_dataset_len, total_val_step)

        # validation step
        fcn.eval()
        total_val_loss = 0  # loss and acc on validation set
        total_val_accuracy = 0

        with torch.no_grad():  # No gradient accumulation for the validation part
            for data in val_loader:
                positions, targets = data
                if torch.cuda.is_available():
                    positions = positions.cuda()
                    targets = targets.cuda()
                outputs = fcn(positions)
                loss = loss_fn(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_val_accuracy = total_val_accuracy + accuracy

        print("Total validation Loss: {}".format(total_val_loss))
        print("Total validation accuracy: {}".format(total_val_accuracy / val_dataset_len))
        writer.add_scalar("Validation_loss", total_val_loss, total_val_step)
        writer.add_scalar("Validation_accuracy", total_val_accuracy / val_dataset_len, total_val_step)
        total_val_step = total_val_step + 1

        # Early Stopping
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_val_acc = total_val_accuracy / val_dataset_len
            torch.save(fcn.state_dict(), f"./State_dict/FCN_State/5fold_1st/Fold_{fold + 1}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(
                    "val_loss has not improved for {} consecutive epoch, early stop at {} round".format(patience_limit,
                                                                                                        total_val_step))
                val_best_loss_5fold.append(best_val_loss)
                val_best_acc_5fold.append(best_val_acc)
                break
    writer.close()
    del fcn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

writer = SummaryWriter("./Logs_tensorboard/FCN_5folds_1st/Average")
avg_val_loss = sum(val_best_loss_5fold) / len(val_best_loss_5fold)
avg_val_acc = sum(val_best_acc_5fold) / len(val_best_acc_5fold)
std_val_loss = std(val_best_loss_5fold)
std_val_acc = std(val_best_acc_5fold)

for i, value in enumerate(val_best_acc_5fold):
    writer.add_scalar("Val_Best_Acc_5fold", value, i)
for i, value in enumerate(val_best_loss_5fold):
    writer.add_scalar("val_best_loss_5fold", value, i)
writer.add_scalar("Mean_val_best_loss", avg_val_loss, 1)
writer.add_scalar("Mean_val_best_accuracy", avg_val_acc, 1)
writer.add_scalar("Std_val_best_loss", std_val_loss, 1)
writer.add_scalar("Std_val_best_accuracy", std_val_acc, 1)
writer.close()
