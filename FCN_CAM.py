import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from Model.FCN import FCN
from ReadData import load_and_process_data, MyDataset, load_single_file_as_dataset


# load origin model
fcn = FCN()
state_path = "./State_dict/Trial1_left_split2/FCN/Fold_3.pth"
fcn.load_state_dict(torch.load(state_path))


# Define a new model same as the original model, but detach feature map from the last convolutional layer
class FCN_CAM(nn.Module):
    def __init__(self):
        super(FCN_CAM, self).__init__()
        self.features = nn.Sequential(*list(fcn.model.children())[:-3])  # to get feature map before gap
        self.gap = list(fcn.model.children())[-3]
        self.flat = list(fcn.model.children())[-2]
        self.fc = list(fcn.model.children())[-1]
        self.feature_map = None

    def forward(self, x):
        x = self.features(x)
        self.feature_map = x.detach()       # detach feature maps from last cl
        x = self.gap(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


# # read data
# train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
#                                                         r"\data_info\trial1_sorted")
# # build dataset
# train_dataset = MyDataset(train_data)
# val_dataset = MyDataset(val_data)
#
# # build dataloader
# batch_size = 16
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#
# fcn_cam = FCN_CAM()     # build model
#
# desired_label = 1  # the class i want
# # iterate over all batches in the data loader
# for datas, labels in train_loader:
#     # iterate over all samples in the batch
#     for i in range(len(datas)):
#         data = datas[i:i+1]  # get the i-th data in the batch
#         label = labels[i]    # get the i-th label in the batch
#         output = fcn_cam(data)     # forward pass through the new model
#
#         # check if the prediction matches the label
#         if output.argmax(1).item() == label.item() == desired_label:
#             label = label.item()
#             break
#     else:
#         # this only executes if the inner loop completed without finding a match
#         continue
#     # this only executes if the inner loop was exited by 'break'
#     break
#
# print(data)
# print(label, output.argmax(1).item())
#
# weights = fcn_cam.fc.weight.detach()    # get the weights of the last linear layer
# weights = weights.transpose(0, 1)  # change the shape of weights from (2, num_channels) to (num_channels, 2)
#
# # calculate cam, channels weighted sum of feature maps
# cam = torch.einsum('ijk,jl->ilk', fcn_cam.feature_map, weights)
#
# # # Apply a ReLu activation
# # cam = nn.functional.relu(cam)
#
# # Percentile normalization
# lower, upper = np.percentile(cam, [50, 99])
# cam = np.clip(cam, lower, upper)  # clip values outside of the lower and upper percentile
# cam = (cam - lower) / (upper - lower)  # scale the values to range [0,1]
#
#
# # # Normalize the CAM to range[0,1]
# # cam = (cam - cam.min()) / (cam.max() - cam.min())
#
# # Create a colormap
# cmap = plt.get_cmap('jet')
#
# # resize data and cam from ([1, 2, 33920]) to (2,33920)
# data = data.squeeze(0)
# cam = cam.squeeze(0)
#
# # # plot the data in one plot
# # fig, ax = plt.subplots(figsize=(10, 6))
# # # Create an array of colors based on the CAM for the given label
# # colors = cmap(cam[label])
# #
# # # Plot the data with the colors
# # for channel in range(data.shape[0]):
# #     for i in range(data.shape[1]-1):
# #         ax.plot(range(i, i+2), data[channel, i:i+2], color=colors[i])
# #
# # plt.show()
#
#
# # plot the data in two subplots
# fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 6))
#
# # Create an array of colors based on the CAM for the given label
# colors = cmap(cam[label])
#
# channel_names = ['abs_x', 'abs_y']
#
# # Plot the data with the colors
# for channel in range(data.shape[0]):
#     for i in range(data.shape[1]-1):
#         axs[channel].plot(range(i, i+2), data[channel, i:i+2], color=colors[i])
#     axs[channel].set_title(channel_names[channel])
#
# plt.tight_layout()
# plt.show()

file_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial1_sorted\0\001_1.csv"
data001_1 = load_single_file_as_dataset(file_path, '0')
train_dataset = MyDataset(data001_1)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

fcn_cam = FCN_CAM()

for sample in train_loader:
    data, label = sample

# print(data.shape)
# print(label)

output = fcn_cam(data)
print(output.argmax(1).item())

weights = fcn_cam.fc.weight.detach()
weights = weights.transpose(0, 1)
cam = torch.einsum('ijk,jl->ilk', fcn_cam.feature_map, weights)

# # Apply a ReLu activation
# cam = nn.functional.relu(cam)

# Percentile normalization
lower, upper = np.percentile(cam, [50, 99])
cam = np.clip(cam, lower, upper)  # clip values outside of the lower and upper percentile
cam = (cam - lower) / (upper - lower)  # scale the values to range [0,1]

# # Normalize the CAM to range[0,1]
# cam = (cam - cam.min()) / (cam.max() - cam.min())

# Create a colormap
cmap = plt.get_cmap('jet')

# resize data and cam from ([1, 2, 33920]) to (2,33920)
data = data.squeeze(0)
cam = cam.squeeze(0)

# plot the data in two subplots
fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 6))
# Create an array of colors based on the CAM for the given label
colors = cmap(cam[0])  # 0类别的话改为0，1的话改为1
channel_names = ['abs_x', 'abs_y']

# Plot the data with the colors
for channel in range(data.shape[0]):
    for i in range(data.shape[1]-1):
        axs[channel].plot(range(i, i+2), data[channel, i:i+2], color=colors[i])
    axs[channel].set_title(channel_names[channel])

plt.tight_layout()
plt.show()

