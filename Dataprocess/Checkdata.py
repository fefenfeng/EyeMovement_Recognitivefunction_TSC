import os
import pandas as pd

"""
Simple Script to detect anomalies in data files.
Especially the data length error may caused by data missing or recording error.
Error data should be excluded in the next experiments.
"""


def detect_anomalies_in_csv_files(folder_path, specified_length):
    anomalous_files = []

    # read all csv files in specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)

            errors = []

            # check data length
            if len(data) != specified_length:
                errors.append(f"Abnormal length: {len(data)} rows, Expected length: {specified_length} rows")

            # Check missing data
            missing_data = data.isnull().sum().sum()  # Compute the number of missed data (missed rows)
            if missing_data > 0:
                errors.append(f"For all, there are {missing_data} missing data (missing rows)")

            if errors:
                anomalous_files.append((file_name, ", ".join(errors)))

    return anomalous_files


folder_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\left_eye\trial0_sorted\0"
specified_length = 40800
# specified_length = 33920
anomalies = detect_anomalies_in_csv_files(folder_path, specified_length)

for anomaly in anomalies:
    print(anomaly)
