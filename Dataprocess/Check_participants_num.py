import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def exponential_moving_average(values, alpha):
    """

    Args:
        values:
        alpha:

    Returns:

    """
    smoothed_values = []
    prev_ema = values[0]
    for value in values:
        ema = alpha * value + (1 - alpha) * prev_ema
        smoothed_values.append(ema)
        prev_ema = ema
    return smoothed_values


def plot_training_loss_with_smoothing(paths, x_label, y_label, legend_labels, colors, smoothing):
    # Create a large figure
    plt.figure(figsize=(12, 6))

    # Loop over paths and plot each one
    for path, label, color in zip(paths, legend_labels, colors):
        df = pd.read_csv(path)
        smoothed_values = exponential_moving_average(df['Value'], smoothing)
        plt.plot(df['Step'], smoothed_values, label=label, color=color, alpha=0.8)

    # Setting labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Validation loss of InceptionTime with different learning rate')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_training_and_validation_loss(train_paths, val_paths, x_label, y_label, legend_labels, colors, smoothing):
    # Create a large figure
    plt.figure(figsize=(24, 6))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    for path, label, color in zip(train_paths, legend_labels, colors):
        df = pd.read_csv(path)
        smoothed_values = exponential_moving_average(df['Value'], smoothing)
        plt.plot(df['Step'], smoothed_values, label=label, color=color, alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plotting validation loss
    plt.subplot(1, 2, 2)
    for path, label, color in zip(val_paths, legend_labels, colors):
        df = pd.read_csv(path)
        smoothed_values = exponential_moving_average(df['Value'], smoothing)
        plt.plot(df['Step'], smoothed_values, label=label, color=color, alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# two line graphs
train_paths = [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\train\run-Only_first_inceptionblock_Fold_1-tag-Train_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\train\run-Only_first_inceptionblock_Fold_2-tag-Train_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\train\run-Only_first_inceptionblock_Fold_3-tag-Train_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\train\run-Only_first_inceptionblock_Fold_4-tag-Train_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\train\run-Only_first_inceptionblock_Fold_5-tag-Train_loss.csv"

]
val_paths = [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\validation\run-Only_first_inceptionblock_Fold_1-tag-Validation_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\validation\run-Only_first_inceptionblock_Fold_2-tag-Validation_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\validation\run-Only_first_inceptionblock_Fold_3-tag-Validation_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\validation\run-Only_first_inceptionblock_Fold_4-tag-Validation_loss.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Transfer_learning\linear+inceptionblock\validation\run-Only_first_inceptionblock_Fold_5-tag-Validation_loss.csv"
]
x_label = 'Epochs'
y_label = 'Loss'
legend_labels = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B']  # Colors in RGB format
smoothing = 0.3  # Adjust this value between 0 and 0.999 for desired smoothing

plot_training_and_validation_loss(train_paths, val_paths, x_label, y_label, legend_labels, colors, smoothing)

# # line graph
# paths = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Hyperparameter_study\InceptionTime\learning_rate\run-nf4_kz(9,19,39)_neck32_lr_1e-4_1e-5-tag-Validation_loss.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Hyperparameter_study\InceptionTime\learning_rate\run-nf4_kz(9,19,39)_neck32_lr_2e-4_2e-5-tag-Validation_loss.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Hyperparameter_study\InceptionTime\learning_rate\run-nf4_kz(9,19,39)_neck32_lr_5e-4_5e-5-tag-Validation_loss.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Hyperparameter_study\InceptionTime\learning_rate\run-nf4_kz(9,19,39)_neck32_lr_1e-3_1e-4-tag-Validation_loss.csv"
# ]
# x_label = 'Epochs'
# y_label = 'Loss'
# # legend_labels = ['bottleneck_4', 'bottleneck_8', 'bottleneck_16', 'bottleneck_32', 'bottleneck_64']
# # legend_labels = ['kl(3,7,15)', 'kl(7,15,31)', 'kl(9,19,39)', 'kl(15,31,63)']
# # legend_labels = ['nf_4', 'nf_8', 'nf_16', 'nf_32']
# legend_labels = ['lr(1e-4 ~ 1e-5)', 'lr(2e-4 ~ 2e-5)', 'lr(5e-4 ~ 5e-5)', 'lr(1e-3 ~ 1e-4)']
# # colors = ['#1F77B4', '#FF7F0E', '#2CA02C']  # Colors in RGB format
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']  # Colors in RGB format
# # colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B']  # Colors in RGB format
# smoothing = 0.3  # Adjust this value between 0 and 0.999 for desired smoothing
#
# plot_training_loss_with_smoothing(paths, x_label, y_label, legend_labels, colors, smoothing)


# Paths to your data files
# paths = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ]
#
# # Read the data
# data_acc = [pd.read_csv(path)['Value'].tolist() for path in paths]
# model_names = ['1DCNN_GAP', 'FCN', 'Resnet', 'InceptionTime']
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']
#
# # Plotting
# plt.figure(figsize=(12, 6))
# bp = plt.boxplot(data_acc, patch_artist=True, vert=True)
#
# # Coloring the boxes
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
#
# plt.xticks(np.arange(1, len(model_names) + 1), model_names)
# plt.title('Boxplot of Best Validation Accuracy for Different Models')
# plt.ylabel('Accuracy')
# plt.grid(axis='y')
#
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define file paths
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define a function to get data from the files
def get_data(acc_files, loss_files):
    acc_data = [pd.read_csv(file)['Value'].tolist() for file in acc_files]
    loss_data = [pd.read_csv(file)['Value'].tolist() for file in loss_files]
    return acc_data, loss_data


# Define a function to calculate mean and standard deviation
def get_statistics(data):
    means = [np.mean(item) for item in data]
    stds = [np.std(item) for item in data]
    return means, stds


# # Define your file paths for four situations
# situations = {
#     'Situation1': {
#         'acc_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ],
#         'loss_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split1\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
# ],
#         'color': '#1F77B4',
#         'legend': 'split1_left_smooth_pursuit'
#     },
#     # Add paths and colors for the other three situations similarly
# 'Situation2': {
#         'acc_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ],
#         'loss_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_left_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
# ],
#         'color': '#FF7F0E',
#         'legend': 'split2_left_smooth_pursuit'
#     },
# 'Situation3': {
#         'acc_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ],
#         'loss_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial1_both_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
# ],
#         'color': '#2CA02C',
#         'legend': 'split2_both_smooth_pursuit'
#     },
# 'Situation4': {
#         'acc_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-1DCNN_GAP_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-FCN_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-Resnet_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\Acc\run-InceptionTime_Average-tag-Val_Best_Acc_5fold.csv"
# ],
#         'loss_files': [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-1DCNN_GAP_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-FCN_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-Resnet_Average-tag-val_best_loss_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Performance_comparison\trial0_left_split2\loss\run-InceptionTime_Average-tag-val_best_loss_5fold.csv"
# ],
#         'color': '#9467BD',
#         'legend': 'split2_left_saccadic_movement'
#     },
# }
#
# # Model names for labeling
# model_names = ['1DCNN_GAP', 'FCN', 'Resnet', 'InceptionTime']
#
# # Plotting
# fig, axs = plt.subplots(1, 2, figsize=(18, 6))
#
# # Loop through each situation and plot
# for situation, details in situations.items():
#     acc_data, loss_data = get_data(details['acc_files'], details['loss_files'])
#     acc_means, acc_stds = get_statistics(acc_data)
#     loss_means, loss_stds = get_statistics(loss_data)
#
#     # Line plot with shaded region for Accuracy
#     axs[0].plot(model_names, acc_means, '-o', color=details['color'], label=details['legend'])
#     axs[0].fill_between(model_names, np.array(acc_means) - np.array(acc_stds), np.array(acc_means) + np.array(acc_stds),
#                         color=details['color'], alpha=0.2)
#
#     # Line plot with shaded region for Loss
#     axs[1].plot(model_names, loss_means, '-o', color=details['color'])
#     axs[1].fill_between(model_names, np.array(loss_means) - np.array(loss_stds),
#                         np.array(loss_means) + np.array(loss_stds), color=details['color'], alpha=0.2)
#
# axs[0].set_title('Mean Validation Accuracy with Standard Deviation')
# axs[0].set_ylabel('Accuracy')
# axs[0].grid(axis='y')
# axs[0].legend(loc='lower right')
#
# axs[1].set_title('Mean Validation Loss with Standard Deviation')
# axs[1].set_ylabel('Loss')
# axs[1].grid(axis='y')
#
# plt.tight_layout()
# plt.show()


# # Load the data
# files = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Step_ramp_tests_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Triangular_waveform_tests_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Random_motion_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Figure_of_eight_tests_Average-tag-Val_Best_Acc_5fold.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Stepped_random_motion_Average-tag-Val_Best_Acc_5fold.csv"
# ]
#
# data = [pd.read_csv(file)['Value'].tolist() for file in files]
#
# # Calculate mean and standard deviation for each dataset
# means = [np.mean(values) for values in data]
# stds = [np.std(values) for values in data]
#
# # Define labels and colors for the bar chart
# labels = [
#     "Step-ramp tests",
#     "Triangular waveform test",
#     "Random motion",
#     "Figure-of-eight tests",
#     "Stepped random motion",
# ]
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B']
#
# # Plotting
# plt.figure(figsize=(12, 6))
# bars = plt.bar(labels, means, yerr=stds, color=colors, capsize=10, alpha=0.8)
#
# # Add title and labels
# plt.title('Different eye movement patterns in smooth pursuit trial')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45)
# # plt.grid(axis='y')
# plt.grid(axis='y', linestyle='--', linewidth=0.5)
#
#
# y_min = min(means) - 0.05
# y_max = max(means) + 0.05
# plt.ylim(y_min, y_max)
#
# plt.tight_layout()
# plt.show()


# # Load the data from the files
# file_paths = [
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Prosaccades_gap_participants_diff-tag-Mean_Accuracy.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Mixed_saccades_gap_participants_diff-tag-Mean_Accuracy.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Random_motion_participants_diff-tag-Mean_Accuracy.csv",
#     r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Step_ramp_tests_participants_diff-tag-Mean_Accuracy.csv"
# ]
#
# dataframes = [pd.read_csv(file_path) for file_path in file_paths]
#
# # Plotting
# plt.figure(figsize=(12, 6))
#
# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']
# labels = ["Prosaccades (gap)", "Mixed saccades (gap)", "Random motion", "Step-ramp tests"]
#
# for df, color, label in zip(dataframes, colors, labels):
#     plt.plot(df['Step'], df['Value'], label=label, color=color)
#
# plt.title('Accuracy over durations (overlap)')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#
# plt.tight_layout()
# plt.show()
