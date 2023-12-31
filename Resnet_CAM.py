import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from Model.ResNet import ResNet
from ReadData import load_single_file_as_dataset, MyDataset

# load origin model
resnet = ResNet()
state_path = "./State_dict/Trial1_left_split2/Resnet/Fold_1.pth"
resnet.load_state_dict(torch.load(state_path))

# print(inceptiontime)

class Resnet_CAM(nn.Module):
    def __init__(self):
        super(Resnet_CAM, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.gap = list(resnet.children())[-3]
        self.flat = list(resnet.children())[-2]
        self.fc = list(resnet.children())[-1]
        self.feature_map = None

    def forward(self, x):
        x = self.features(x)
        self.feature_map = x.detach()
        x = self.gap(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

file_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial1_sorted\0\001_1.csv"
data001_1 = load_single_file_as_dataset(file_path, '0')
train_dataset = MyDataset(data001_1)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

resnet_cam = Resnet_CAM()

for sample in train_loader:
    data, label = sample

# print(data.shape)
# print(label)

output = resnet_cam(data)
print(output.argmax(1).item())

weights = resnet_cam.fc.weight.detach()
weights = weights.transpose(0, 1)
cam = torch.einsum('ijk,jl->ilk', resnet_cam.feature_map, weights)

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
