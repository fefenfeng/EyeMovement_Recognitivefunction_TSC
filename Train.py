import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from ReadData import load_and_process_data, MyDataset
from CNN1d import CNN1d
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
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# networks model instantiation
cnn1d = CNN1d()
if torch.cuda.is_available():
    cnn1d = cnn1d.cuda()    # 转移到cuda上
# define loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# define optimizer
learning_rate = 1e-3
l2penalty = 1e-3
optimizer = torch.optim.Adam(cnn1d.parameters(), lr=learning_rate, weight_decay=l2penalty)

# Set up for some parameters in training
total_train_step = 0   # 训练次数
total_val_step = 0   # 测试次数
epoch = 100     # 训练轮数

# early stopping
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 10

# tensorboard
writer = SummaryWriter("./Logs_tensorboard/CNN1d_1st_train")
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # training begin
    cnn1d.train()   # turn to training mode
    for data in train_loader:
        positions, targets = data  # feature+label-->data
        if torch.cuda.is_available():
            positions = positions.cuda()
            targets = targets.cuda()
        outputs = cnn1d(positions)
        loss = loss_fn(outputs, targets)  # calculate loss
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
        if total_train_step % 50 == 0:
            end_time = time.time()
            train_time = end_time - start_time
            print("Total train step: {}, Train time: {}, Loss: {}".format(total_train_step, train_time, loss.item()))
            writer.add_scalar("Train_loss", loss.item(), total_train_step)

    # validation step
    cnn1d.eval()
    total_val_loss = 0  # loss and acc on validation set
    total_val_accuracy = 0
    with torch.no_grad():   # No gradient accumulation for the validation part
        for data in val_loader:
            positions, targets = data
            if torch.cuda.is_available():
                positions = positions.cuda()
                targets = targets.cuda()
            outputs = cnn1d(positions)
            loss = loss_fn(outputs, targets)
            total_val_loss = total_val_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_val_accuracy = total_val_accuracy + accuracy

    print("Total validation Loss: {}".format(total_val_loss))
    print("Total validation accuracy: {}".format(total_val_accuracy/val_dataset_len))
    writer.add_scalar("Validation_loss", total_val_loss, total_val_step)
    writer.add_scalar("Validation_accuracy", total_val_accuracy/val_dataset_len, total_val_step)
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
