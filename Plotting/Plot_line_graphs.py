import pandas as pd
import matplotlib.pyplot as plt


def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average (EMA) of a list of values.
    This is for smoothing the acc/loss curves.

    Args:
        values (list of float): A list of float values for which to compute the EMA.
        alpha (float): The smoothing factor, between 0 and 1. A smaller alpha, more smoothing.

    Returns:
        list of float:
        A list containing smoothed values.
    """
    smoothed_values = []
    prev_ema = values[0]
    for value in values:
        ema = alpha * value + (1 - alpha) * prev_ema
        smoothed_values.append(ema)
        prev_ema = ema
    return smoothed_values


def plot_training_loss_with_smoothing(paths, x_label, y_label, legend_labels, colors, smoothing):
    """
    Plot training loss/acc curves with EMA smoothing.

    Args:
        paths (list of str): Paths to CSV files containing training loss data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        legend_labels (list of str): Labels for each curve in the legend.
        colors (list of str): Colors for each curve.
        smoothing (float): Smoothing factor for EMA smooth.

    Returns:
        None:
        Plot training loss curves of results of folds cross validation.
    """
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
    """
    Plot both training and validation loss/acc curves with EMA smoothing.

    Args:
        train_paths (list of str): Paths to CSV files containing training loss data.
        val_paths (list of str): Paths to CSV files containing validation loss data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        legend_labels (list of str): Labels for each curve in the legend.
        colors (list of str): Colors for each curve.
        smoothing (float): Smoothing factor.

    Returns:
        None:
        Plot both train and validation loss curves in one figure.
    """
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


# two line graphs example
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

# # single line graph exmaple
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
