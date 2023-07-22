import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from ReadData import load_and_process_data, MyDataset
# from Model.CNN1d import CNN1d
from Model.CNN1d_GAP import CNN1d_GAP
import time

# read data
train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
                                                        r"\data_info\trial1_sorted")
# build dataset
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
# test_dataset = MyDataset(test_data)

# dataset length
train_dataset_len = len(train_dataset)
print("The length of train dataset is :{}".format(train_dataset_len))
val_dataset_len = len(val_dataset)
print("The length of validation dataset is :{}".format(val_dataset_len))
# test_dataset_len = len(test_dataset)
# print("The length of test dataset is:{}".format(test_dataset_len))

# build dataloader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# networks model instantiation
cnn1d_gap = CNN1d_GAP()
if torch.cuda.is_available():
    cnn1d_gap = cnn1d_gap.cuda()    # 转移到cuda上
# define loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# define optimizer
learning_rate = 5e-5
# l2penalty = 1e-3
# optimizer = torch.optim.Adam(cnn1d_gap.parameters(), lr=learning_rate, weight_decay=l2penalty)  # add L2 regularization
optimizer = torch.optim.Adam(cnn1d_gap.parameters(), lr=learning_rate)
# Set up for some parameters in training
total_train_step = 0   # 训练次数
total_val_step = 0   # 测试次数
epoch = 1000     # 训练轮数

# early stopping
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 50

# tensorboard
writer = SummaryWriter("./Logs_tensorboard/CNN1d_GAP_3rd")
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # training begin
    cnn1d_gap.train()   # turn to training mode
    total_train_loss = 0
    total_train_accuracy = 0
    for data in train_loader:
        positions, targets = data  # feature+label-->data
        if torch.cuda.is_available():
            positions = positions.cuda()
            targets = targets.cuda()
        outputs = cnn1d_gap(positions)
        loss = loss_fn(outputs, targets)  # calculate loss
        total_train_loss = total_train_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_train_accuracy = total_train_accuracy + accuracy
        optimizer.zero_grad()   # turn optimizer gradient--> zero
        loss.backward()     # backward propagation
        # if hasattr(torch.cuda, 'empty_cache'):    # to avoid cuda out of memory
        #     torch.cuda.empty_cache()
        #     print("empty_cache")
        optimizer.step()    # Step gradient update
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        #     print("empty_cache")

        total_train_step = total_train_step + 1
        # train loss step
        if total_train_step % 50 == 0:
            end_time = time.time()
            train_time = end_time - start_time
            print("Total train step: {}, Train time: {}, Loss: {}".format(total_train_step, train_time, loss.item()))
            writer.add_scalar("Train_loss_step_local", loss.item(), total_train_step)

    # total train loss and acc
    print("Total train Loss: {}".format(total_train_loss))
    print("Total train accuracy: {}".format(total_train_accuracy/train_dataset_len))
    writer.add_scalar("Train_loss_local", total_train_loss, i+1)
    writer.add_scalar("Train_accuracy_local", total_train_accuracy/train_dataset_len, i+1)

    # validation step
    cnn1d_gap.eval()
    total_val_loss = 0  # loss and acc on validation set
    total_val_accuracy = 0
    with torch.no_grad():   # No gradient accumulation for the validation part
        for data in val_loader:
            positions, targets = data
            if torch.cuda.is_available():
                positions = positions.cuda()
                targets = targets.cuda()
            outputs = cnn1d_gap(positions)
            loss = loss_fn(outputs, targets)
            total_val_loss = total_val_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_val_accuracy = total_val_accuracy + accuracy

    print("Total validation Loss: {}".format(total_val_loss))
    print("Total validation accuracy: {}".format(total_val_accuracy/val_dataset_len))
    writer.add_scalar("Validation_loss_local", total_val_loss, total_val_step)
    writer.add_scalar("Validation_accuracy_local", total_val_accuracy/val_dataset_len, total_val_step)
    total_val_step = total_val_step + 1

    # Early Stopping
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print("val_loss has not improved for {} consecutive epoch, early stop at {} round".format(patience_limit, total_val_step))
            break
writer.close()
