import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Plot line graphs with shade area representing the error area.
Show both mean and standard value. Show results' distributions.
Mean(line): performance expectations 
Distributions(shades): performance stability
"""


def get_data(acc_files, loss_files):
    acc_data = [pd.read_csv(file)['Value'].tolist() for file in acc_files]
    loss_data = [pd.read_csv(file)['Value'].tolist() for file in loss_files]
    return acc_data, loss_data


def get_statistics(data):
    means = [np.mean(item) for item in data]
    stds = [np.std(item) for item in data]
    return means, stds


# Set CSV results files path, color and legend for each situation (situation means trial)
situations = {
    'Situation1': {
        'acc_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
],
        'loss_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
],
        'color': '#1F77B4',
        'legend': 'split1_left_smooth_pursuit'
    },
'Situation2': {
        'acc_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
],
        'loss_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
],
        'color': '#FF7F0E',
        'legend': 'split2_left_smooth_pursuit'
    },
'Situation3': {
        'acc_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
],
        'loss_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
],
        'color': '#2CA02C',
        'legend': 'split2_both_smooth_pursuit'
    },
'Situation4': {
        'acc_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
],
        'loss_files': [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
],
        'color': '#9467BD',
        'legend': 'split2_left_saccadic_movement'
    },
}

# Model names for labeling
model_names = ['1DCNN_GAP', 'FCN', 'Resnet', 'InceptionTime']

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Loop through each situation and plot
for situation, details in situations.items():
    acc_data, loss_data = get_data(details['acc_files'], details['loss_files'])
    acc_means, acc_stds = get_statistics(acc_data)
    loss_means, loss_stds = get_statistics(loss_data)

    # Line plot with shaded region for Accuracy
    axs[0].plot(model_names, acc_means, '-o', color=details['color'], label=details['legend'])
    axs[0].fill_between(model_names, np.array(acc_means) - np.array(acc_stds), np.array(acc_means) + np.array(acc_stds),
                        color=details['color'], alpha=0.2)

    # Line plot with shaded region for Loss
    axs[1].plot(model_names, loss_means, '-o', color=details['color'])
    axs[1].fill_between(model_names, np.array(loss_means) - np.array(loss_stds),
                        np.array(loss_means) + np.array(loss_stds), color=details['color'], alpha=0.2)

axs[0].set_title('Mean Validation Accuracy with Standard Deviation')
axs[0].set_ylabel('Accuracy')
axs[0].grid(axis='y')
axs[0].legend(loc='lower right')

axs[1].set_title('Mean Validation Loss with Standard Deviation')
axs[1].set_ylabel('Loss')
axs[1].grid(axis='y')

plt.tight_layout()
plt.show()


# # Single trial's 4 models' acc and loss: mean value with std represented by line graphs with shade
# acc_files = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ]
#
# loss_files = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
# ]
#
# # Extract data from the files
# acc_data = [pd.read_csv(file)['Value'].tolist() for file in acc_files]
# loss_data = [pd.read_csv(file)['Value'].tolist() for file in loss_files]
#
# # Calculate mean and standard deviation for accuracy and loss
# acc_means = [np.mean(data) for data in acc_data]
# acc_stds = [np.std(data) for data in acc_data]
# loss_means = [np.mean(data) for data in loss_data]
# loss_stds = [np.std(data) for data in loss_data]
#
# # Model names for labeling
# model_names = ['1DCNN_GAP', 'FCN', 'Resnet', 'InceptionTime']
#
# # Plotting
# fig, axs = plt.subplots(1, 2, figsize=(18, 6))
#
# # Line plot with shaded region for Accuracy
# axs[0].plot(model_names, acc_means, '-o', color='#1F77B4', label='Mean Accuracy')
# axs[0].fill_between(model_names, np.array(acc_means) - np.array(acc_stds), np.array(acc_means) + np.array(acc_stds), color='#1F77B4', alpha=0.2)
# axs[0].set_title('Mean Validation Accuracy with Standard Deviation')
# axs[0].set_ylabel('Accuracy')
# axs[0].grid(axis='y')
#
# # Line plot with shaded region for Loss
# axs[1].plot(model_names, loss_means, '-o', color='#1F77B4', label='Mean Loss')
# axs[1].fill_between(model_names, np.array(loss_means) - np.array(loss_stds), np.array(loss_means) + np.array(loss_stds), color='#1F77B4', alpha=0.2)
# axs[1].set_title('Mean Validation Loss with Standard Deviation')
# axs[1].set_ylabel('Loss')
# axs[1].grid(axis='y')
#
# plt.tight_layout()
# plt.show()
