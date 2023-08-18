import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from Model.InceptionTime import InceptionTime
from ReadData import load_single_file_as_dataset, MyDataset, load_all_data

class InceptionTime_CAM(nn.Module):
    def __init__(self, inceptiontime):
        super(InceptionTime_CAM, self).__init__()
        self.features = nn.Sequential(*list(inceptiontime.model.children())[:-3])
        self.gap = list(inceptiontime.model.children())[-3]
        self.flat = list(inceptiontime.model.children())[-2]
        self.fc = list(inceptiontime.model.children())[-1]
        self.feature_map = None

    def forward(self, x):
        x = self.features(x)
        self.feature_map = x.detach()
        x = self.gap(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

def calculate_mean_cam(weight_path):
    inceptiontime = InceptionTime()
    inceptiontime.load_state_dict(torch.load(weight_path))
    inceptiontime_cam = InceptionTime_CAM(inceptiontime)
    all_cam = torch.zeros((2, 33920))
    for sample in all_loader:
        data, label = sample
        output = inceptiontime_cam(data)
        weights = inceptiontime_cam.fc.weight.detach()
        weights = weights.transpose(0, 1)
        cam = torch.einsum('ijk,jl->ilk', inceptiontime_cam.feature_map, weights)
        all_cam += cam.mean(dim=0)
    return all_cam / len(all_loader)

# load all data and calculate mean CAM value
folder_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial1_sorted"
alldata = load_all_data(folder_path)
all_dataset = MyDataset(alldata)
batch_size = 16
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

# Paths for the five weights from the cross-validation
weight_paths = [
    "./State_dict/Trial1_left_split2/InceptionTime/Fold_1.pth",
    "./State_dict/Trial1_left_split2/InceptionTime/Fold_2.pth",
    "./State_dict/Trial1_left_split2/InceptionTime/Fold_3.pth",
    "./State_dict/Trial1_left_split2/InceptionTime/Fold_4.pth",
    "./State_dict/Trial1_left_split2/InceptionTime/Fold_5.pth"]
# Calculate the mean_cam for each weight and sum them
total_mean_cam = torch.zeros((2, 33920))
for weight_path in weight_paths:
    total_mean_cam += calculate_mean_cam(weight_path)
final_mean_cam = total_mean_cam / 5

# Percentile normalization
lower, upper = np.percentile(final_mean_cam, [50, 99])
final_mean_cam = np.clip(final_mean_cam, lower, upper)  # clip values outside of the lower and upper percentile
final_mean_cam = (final_mean_cam - lower) / (upper - lower)  # scale the values to range [0,1]

# single file read
file_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial1_sorted\0\001_1.csv"
data001_1 = load_single_file_as_dataset(file_path, '0')
train_dataset = MyDataset(data001_1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for sample in train_loader:
    data, label = sample
# Create a colormap
cmap = plt.get_cmap('jet')

# resize data and cam from ([1, 2, 33920]) to (2,33920)
data = data.squeeze(0)
# cam = cam.squeeze(0)

# plot the data in two subplots
fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 6))
# Create an array of colors based on the CAM for the given label
colors = cmap(final_mean_cam[0])  # 0类别的话改为0，1的话改为1
channel_names = ['abs_x', 'abs_y']

# Plot the data with the colors
for channel in range(data.shape[0]):
    for i in range(data.shape[1]-1):
        axs[channel].plot(range(i, i+2), data[channel, i:i+2], color=colors[i])
    axs[channel].set_title(channel_names[channel])

plt.tight_layout()
plt.show()

# # CAM over each pattern stages
# # Given index ranges
# # index_ranges = [[2240, 8800], [8800, 15200], [15200, 21440], [21440, 28000], [28000, 40800]]
# index_ranges = [[2240, 14400], [14400, 20480], [20480, 24000], [24000, 26400], [26400, 33920]]
# # Calculate the average values for each index range
# average_values = []
# for start, end in index_ranges:
#     segment_mean = final_mean_cam[:, start:end].mean(dim=1)
#     average_values.append(segment_mean.unsqueeze(1))
#
# # Concatenate all average values to get a result of shape (2, len(index_ranges))
# average_values_tensor = torch.cat(average_values, dim=1)
# print(average_values_tensor)
#
# # Given range names
# # range_names = ["Prosaccade(gap)", "Prosaccade(overlap)", "Antisaccade(gap)", "Antisaccade(overlap)", "Mixed Saccade(gap)"]
# range_names = ["Step-ramp tests", "Triangular waveform test", "Random motion", "Figure-of-eights", "Stepped random motion"]
# # Plotting the bar chart
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#
# for i, title in enumerate(["No cognitive load", "Under cognitive load"]):
#     bars = axs[i].bar(range_names, average_values_tensor[i].numpy(), color=['blue', 'green', 'red', 'cyan', 'purple'])
#     axs[i].set_title(title)
#     axs[i].set_ylabel("Average Normalized CAM Value")
#     axs[i].set_xticks(range(len(range_names)))
#     # axs[i].set_xticklabels(range_names, rotation=45, ha='right')
#     axs[i].set_xticklabels(range_names)
#
#     for bar in bars:
#         yval = bar.get_height()
#         axs[i].text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 4), ha='center', va='bottom',
#                     fontsize=10)
#
# plt.tight_layout()
# plt.show()

# # Mean CAM over all data
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
# axs[0].plot(final_mean_cam[0].numpy())
# axs[0].set_title("No cognitive load")
# axs[1].plot(final_mean_cam[1].numpy())
# axs[1].set_title("Under cognitive load")
# plt.tight_layout()
# plt.show()



# # load origin model
# inceptiontime = InceptionTime()
# state_path = "./State_dict/Trial0_left_split2/InceptionTime/Fold_5.pth"
# inceptiontime.load_state_dict(torch.load(state_path))
# # print(inceptiontime)
#
# inceptiontime_cam = InceptionTime_CAM()

# all_cam = torch.zeros((2, 40800))
# for sample in all_loader:
#     data, label = sample
#     output = inceptiontime_cam(data)
#     weights = inceptiontime_cam.fc.weight.detach()
#     weights = weights.transpose(0, 1)
#     cam = torch.einsum('ijk,jl->ilk', inceptiontime_cam.feature_map, weights)
#     all_cam += cam.mean(dim=0)
#
# mean_cam = all_cam / len(all_loader)
# Percentile normalization
# lower, upper = np.percentile(mean_cam, [50, 99])
# mean_cam = np.clip(mean_cam, lower, upper)  # clip values outside of the lower and upper percentile
# mean_cam = (mean_cam - lower) / (upper - lower)  # scale the values to range [0,1]

# # CAM over each pattern stages
# # Given index ranges
# index_ranges = [[2240, 8800], [8800, 15200], [15200, 21440], [21440, 28000], [28000, 40800]]
#
# # Calculate the average values for each index range
# average_values = []
# for start, end in index_ranges:
#     segment_mean = mean_cam[:, start:end].mean(dim=1)
#     average_values.append(segment_mean.unsqueeze(1))
#
# # Concatenate all average values to get a result of shape (2, len(index_ranges))
# average_values_tensor = torch.cat(average_values, dim=1)
# print(average_values_tensor)
#
# # Given range names
# range_names = ["Prosaccade(gap)", "Prosaccade(overlap)", "Antisaccade(gap)", "Antisaccade(overlap)", "Mixed Saccade(gap)"]
#
# # Plotting the bar chart
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#
# for i, title in enumerate(["No cognitive load", "Under cognitive load"]):
#     bars = axs[i].bar(range_names, average_values_tensor[i].numpy(), color=['blue', 'green', 'red', 'cyan', 'purple'])
#     axs[i].set_title(title)
#     axs[i].set_ylabel("Average Normalized CAM Value")
#     axs[i].set_xticks(range(len(range_names)))
#     # axs[i].set_xticklabels(range_names, rotation=45, ha='right')
#     axs[i].set_xticklabels(range_names)
#
#     for bar in bars:
#         yval = bar.get_height()
#         axs[i].text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 4), ha='center', va='bottom',
#                     fontsize=10)
#
# plt.tight_layout()
# plt.show()

# # Mean CAM over all data
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
# axs[0].plot(mean_cam[0].numpy())
# axs[0].set_title("No cognitive load")
# axs[1].plot(mean_cam[1].numpy())
# axs[1].set_title("Under cognitive load")
# plt.tight_layout()
# plt.show()

# # single file read
# file_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial0_sorted\1\001_2.csv"
# data001_1 = load_single_file_as_dataset(file_path, '0')
# train_dataset = MyDataset(data001_1)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# for sample in train_loader:
#     data, label = sample
#
# output = inceptiontime_cam(data)
# print(output.argmax(1).item())
#
# # weights = inceptiontime_cam.fc.weight.detach()
# # weights = weights.transpose(0, 1)
# # cam = torch.einsum('ijk,jl->ilk', inceptiontime_cam.feature_map, weights)
# #
# # # # Apply a ReLu activation
# # # cam = nn.functional.relu(cam)
# #
# # # Percentile normalization
# # lower, upper = np.percentile(cam, [50, 99])
# # cam = np.clip(cam, lower, upper)  # clip values outside of the lower and upper percentile
# # cam = (cam - lower) / (upper - lower)  # scale the values to range [0,1]
# #
# # # # Normalize the CAM to range[0,1]
# # # cam = (cam - cam.min()) / (cam.max() - cam.min())
# #
# # Create a colormap
# cmap = plt.get_cmap('jet')
#
# # resize data and cam from ([1, 2, 33920]) to (2,33920)
# data = data.squeeze(0)
# # cam = cam.squeeze(0)
#
# # plot the data in two subplots
# fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 6))
# # Create an array of colors based on the CAM for the given label
# colors = cmap(mean_cam[1])  # 0类别的话改为0，1的话改为1
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
