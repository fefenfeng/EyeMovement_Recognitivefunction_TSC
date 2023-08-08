import os
import pandas as pd
# from collections import Counter
#
#
# # check abnormal length data file
# def check_csv_len(directory):
#     len_counts = {}
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             df = pd.read_csv(os.path.join(directory, filename))
#             len_counts[filename] = len(df)
#
#     count_counter = Counter(len_counts.values())  # 创建counter对象用于记数行数
#     most_common_len = count_counter.most_common(1)[0][0]  # most_common(1)为列表，包含元组（行数，次数）
#
#     for filename, len_count in len_counts.items():
#         if len_count != most_common_len:
#             print(f'File {filename} has {len_count} rows, which is different from the most common length.')
#
#
# check_csv_len(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\both_eyes\trial0_sorted\1")


def detect_anomalies_in_csv_files(folder_path, specified_length):
    anomalous_files = []

    # 遍历文件夹，读取所有CSV文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)

            errors = []

            # 检查文件长度是否与指定长度相同
            if len(data) != specified_length:
                errors.append(f"异常长度: {len(data)} 行, 期望长度: {specified_length} 行")

            # 检查缺失数据
            missing_data = data.isnull().sum().sum()  # 计算整个数据框中的缺失值数量
            if missing_data > 0:
                errors.append(f"存在 {missing_data} 个缺失数据")

            if errors:
                anomalous_files.append((file_name, ", ".join(errors)))

    return anomalous_files


folder_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\left_eye\trial0_sorted\0"
specified_length = 40800
# specified_length = 33920
anomalies = detect_anomalies_in_csv_files(folder_path, specified_length)

for anomaly in anomalies:
    print(anomaly)
