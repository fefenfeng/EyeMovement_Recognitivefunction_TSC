import os
import pandas as pd

"""
Segment the sequences into subsequences according to specified indices.
This is for research on a specified part of the whole data sequence.
eg. We can segment the data with indices [0,1600], i.e. [0s, 10s].
As this way, we can locate the data in a specific eye-movement phase.
"""


def process_files(source_folder, target_folder, indices):
    # check if target folder is existed, if not create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(source_folder, file_name)
            df = pd.read_csv(file_path)

            # slice the data with indices
            subset = df.iloc[indices[0]:indices[1]]

            # 保存到目标文件夹中
            subset.to_csv(os.path.join(target_folder, file_name), index=False)


# Example
source_folder = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial1\Stepped_random_motion\1"
target_folder = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial1\Stepped_random_motion_3\1"

indices = [4800, 7200]  # indices for the subset
process_files(source_folder, target_folder, indices)

source_folder1 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial1\Stepped_random_motion\0"
target_folder1 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial1\Stepped_random_motion_3\0"
process_files(source_folder1, target_folder1, indices)
