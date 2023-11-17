import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
Plot boxes plots showing the performance comparison between different models in single trial.
Not used in eventual paper figures.
Cause we use 5-folds, number of samples for one trial is just 5,
too small that boxplot may mis-recognize a normal data as a error point.
"""

# Trying if boxes plot can demonstrate the performance comparison between different models in single trial
paths = [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
]

data_acc = [pd.read_csv(path)['Value'].tolist() for path in paths]
model_names = ['1DCNN_GAP', 'FCN', 'Resnet', 'InceptionTime']
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']

plt.figure(figsize=(12, 6))
bp = plt.boxplot(data_acc, patch_artist=True, vert=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.xticks(np.arange(1, len(model_names) + 1), model_names)
plt.title('Boxplot of Best Validation Accuracy for Different Models')
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


