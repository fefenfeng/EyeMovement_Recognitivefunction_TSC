import os
import pandas as pd

"""
Resort the data by time and delete unused column.
"""


def process_files(source_dir, target_dir):
    # Get all csv files in source directory
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

    for file in csv_files:
        df = pd.read_csv(os.path.join(source_dir, file))

        # # get time tag from the 'filename' column
        # df['sequence'] = df['filename'].str.extract('-(\d+)\.jpg')[0]
        #
        # # check if NaN exists
        # if df['sequence'].isna().any():
        #     print(f"NaN value found in file: {file}")
        #     df['sequence'] = df['sequence'].fillna('some_value')  # replace 'some_value'
        #
        # # transform 'sequence' column into int type
        # df['sequence'] = df['sequence'].astype(int)
        #
        # # resort by 'sequence' column
        # df = df.sort_values('sequence')

        # # drop 'filename', 'Unnamed: 0', 'sequence' columns
        # df = df.drop(['Unnamed: 0', 'filename', 'sequence'], axis=1)

        df = df.drop(['Unnamed: 0', 'filename', 'blink'], axis=1)

        # save the data into target directory
        df.to_csv(os.path.join(target_dir, file), index=False)


source_dir = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\left_eye\trial1\0"
target_dir = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\left_eye\trial1_sorted\0"
process_files(source_dir, target_dir)
