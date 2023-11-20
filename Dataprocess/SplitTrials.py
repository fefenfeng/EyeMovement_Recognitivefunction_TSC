import os
import shutil

"""
Split trials into different folders"
Four situations:
trial0: smooth pursuit with low cognitive load
trial1: saccadic movements with low cognitive load
trial2: smooth pursuit with high cognitive load
trial3: saccadic movements with high cognitive load
"""

folder_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\eye_centre_dataset_inc_blinks\right_eye"
folder_0 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial0\0"
folder_1 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial1\0"
folder_2 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial0\1"
folder_3 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial1\1"

folders = [folder_0, folder_1, folder_2, folder_3]

# counter initialization
counts = [0, 0, 0, 0]

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        name_without_ext, ext = os.path.splitext(filename)
        num = int(name_without_ext.split('_')[-1])

        shutil.copy(os.path.join(folder_path, filename), folders[num])

        counts[num] += 1

for i, count in enumerate(counts):
    print(f"The number of data in Trial_{i} are: {count}")


