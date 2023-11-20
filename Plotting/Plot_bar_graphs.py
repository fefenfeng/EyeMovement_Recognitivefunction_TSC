import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Plot bar charts with error (standard deviation) area.
Used to show a particular model's accuracy over different eye movements data in trials
(smooth pursuit / saccadic movements).
"""

# trial0, different movements phases in smooth pursuit trial
files = [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Step_ramp_tests_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Triangular_waveform_tests_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Random_motion_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Figure_of_eight_tests_Average-tag-Val_Best_Acc_5fold.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\Eye_movement_patterns\trial1\run-Stepped_random_motion_Average-tag-Val_Best_Acc_5fold.csv"
]

data = [pd.read_csv(file)['Value'].tolist() for file in files]

# Calculate mean and standard deviation for each dataset
means = [np.mean(values) for values in data]
stds = [np.std(values) for values in data]

# Define labels and colors for the bar chart
labels = [
    "Step-ramp tests",
    "Triangular waveform test",
    "Random motion",
    "Figure-of-eight tests",
    "Stepped random motion",
]
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B']

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, means, yerr=stds, color=colors, capsize=10, alpha=0.8)

# Add title and labels
plt.title('Different eye movement patterns in smooth pursuit trial')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
# plt.grid(axis='y')
plt.grid(axis='y', linestyle='--', linewidth=0.5)

y_min = min(means) - 0.05
y_max = max(means) + 0.05
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()



