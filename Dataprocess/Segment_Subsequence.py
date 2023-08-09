import os
import pandas as pd


def process_files(source_folder, target_folder, indices):
    # 判断目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历原文件夹中的所有CSV文件
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            # 读取CSV文件
            file_path = os.path.join(source_folder, file_name)
            df = pd.read_csv(file_path)

            # 根据子序列切分索引获取数据
            subset = df.iloc[indices[0]:indices[1]]

            # 保存到目标文件夹中
            subset.to_csv(os.path.join(target_folder, file_name), index=False)


# 示例用法
source_folder = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\original_data\trial0_sorted\1"
target_folder = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial0\Mixed_saccades_gap\1"
indices = [28000, 40800]  # 子序列切分索引
process_files(source_folder, target_folder, indices)
