import os
import pandas as pd
from collections import Counter


def check_csv_len(directory):
    len_counts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            len_counts[filename] = len(df)

    count_counter = Counter(len_counts.values())  # 创建counter对象用于记数行数
    most_common_len = count_counter.most_common(1)[0][0]  # most_common(1)为列表，包含元组（行数，次数）

    for filename, len_count in len_counts.items():
        if len_count != most_common_len:
            print(f'File {filename} has {len_count} rows, which is different from the most common length.')


check_csv_len(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial0_sorted\1")