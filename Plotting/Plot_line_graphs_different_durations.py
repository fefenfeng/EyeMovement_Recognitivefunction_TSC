import pandas as pd
import matplotlib.pyplot as plt

"""
Plot line graphs with different durations.
Plot model's accuracies of different eye-movement phase over different time durations.
Look for what time is enough for recognizing the cognitive load change.
And look for what eye-movement might be the best chosen for detecting cognitive load change.
"""

# Load the data from the files
file_paths = [
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Prosaccades_gap_participants_diff-tag-Mean_Accuracy.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Mixed_saccades_gap_participants_diff-tag-Mean_Accuracy.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Random_motion_participants_diff-tag-Mean_Accuracy.csv",
    r"D:\MyFiles\UOB_Robotics22\Dissertation\Results\data_duration\overlap\run-Step_ramp_tests_participants_diff-tag-Mean_Accuracy.csv"
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Plotting
plt.figure(figsize=(12, 6))

colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']
labels = ["Prosaccades (gap)", "Mixed saccades (gap)", "Random motion", "Step-ramp tests"]

for df, color, label in zip(dataframes, colors, labels):
    plt.plot(df['Step'], df['Value'], label=label, color=color)

plt.title('Accuracy over durations (overlap)')
plt.xlabel('Duration (seconds)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
