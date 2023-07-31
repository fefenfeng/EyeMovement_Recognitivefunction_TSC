import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from Model.FCN import FCN
from ReadData import load_and_process_data, MyDataset


# load origin model
fcn = FCN()
state_path = "./State_dict/FCN_State/FineTuning_500epoch/nf_(64,128,64),kz_(7,5,3).pth"
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


# read data
train_data, val_data, test_data = load_and_process_data(r"D:\MyFiles\UOB_Robotics22\Dissertation"
                                                        r"\data_info\trial1_sorted")
# build dataset
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

# build dataloader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

data_iter = iter(train_loader)
first_batch = next(data_iter)   # first batch in loader
datas, labels = first_batch
data = datas[0:1]       # first data in first batch
label = labels[0:1]
# print(label)
label = label.item()
print(label)
# label = 0

fcn_cam = FCN_CAM()
output = fcn_cam(data)  # forward pass through the new model

weights = fcn_cam.fc.weight.detach()    # get the weights of the last linear layer
weights = weights.transpose(0, 1)  # change the shape of weights from (2, num_channels) to (num_channels, 2)

# calculate cam, channels weighted sum of feature maps
cam = torch.einsum('ijk,jl->ilk', fcn_cam.feature_map, weights)

# Apply a ReLu activation
cam = nn.functional.relu(cam)

# Normalize the CAM to range[0,1]
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Create a colormap
cmap = plt.get_cmap('jet')

# resize data and cam from ([1, 2, 33920]) to (2,33920)
data = data.squeeze(0)
cam = cam.squeeze(0)

# plot the data
fig, ax = plt.subplots(figsize=(10, 6))
# Create an array of colors based on the CAM for the given label
colors = cmap(cam[label])

# Plot the data with the colors
for channel in range(data.shape[0]):
    for i in range(data.shape[1]-1):
        ax.plot(range(i, i+2), data[channel, i:i+2], color=colors[i])

plt.show()


# # plot the data
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

