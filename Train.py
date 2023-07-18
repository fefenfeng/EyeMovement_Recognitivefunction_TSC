import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from ReadData import load_and_process_data, MyDataset
from CNN1d import CNN1d
import time

# 读取总体数据
train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
                                                        r"\data_info\trial1_sorted")
# 创建dataset
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
# test_dataset = MyDataset(test_data)

# dataset length
train_dataset_len = len(train_dataset)
print("训练数据集长度为:{}".format(train_dataset_len))
val_dataset_len = len(val_dataset)
print("验证数据集长度为:{}".format(val_dataset_len))
# test_dataset_len = len(test_dataset)
# print("测试数据集长度为:{}".format(test_dataset_len))

# 创建dataloader
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 创建网络模型
cnn1d = CNN1d()
if torch.cuda.is_available():
    cnn1d = cnn1d.cuda()    # 转移到cuda上
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(cnn1d.parameters(), lr=learning_rate)

# 设置训练的一些参数
total_train_step = 0   # 训练次数
total_val_step = 0   # 测试次数
epoch = 100     # 训练轮数

# early stopping设置
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 10

# tensorboard记录
writer = SummaryWriter("./Logs_tensorboard/CNN1d_1st_train")
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # 训练开始
    cnn1d.train()   # 转为训练模式
    for data in train_loader:
        positions, targets = data  # 取feature和label
        if torch.cuda.is_available():
            positions = positions.cuda()
            targets = targets.cuda()
        outputs = cnn1d(positions)
        loss = loss_fn(outputs, targets)  # 计算loss
        optimizer.zero_grad()   # 优化器梯度清零
        loss.backward()     # 反向传播
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        #     print("empty_cache")
        optimizer.step()    # 梯度更新
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        #     print("empty_cache")

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            end_time = time.time()
            train_time = end_time - start_time
            print("训练次数: {}, 训练时长: {}, Loss: {}".format(total_train_step, train_time, loss.item()))
            writer.add_scalar("Train_loss", loss.item(), total_train_step)

    # 验证步骤
    cnn1d.eval()
    total_val_loss = 0  # 验证过程中累计的loss和acc
    total_val_accuracy = 0
    with torch.no_grad():   # 验证部分不累计梯度
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

    print("整体验证集上Loss: {}".format(total_val_loss))
    print("整体验证集上accuracy: {}".format(total_val_accuracy/val_dataset_len))
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
            print("val_loss已经连续{}轮没有改进了,在第{}轮早停".format(patience_limit, total_val_step))
            break
writer.close()
