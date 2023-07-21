import os
import pandas as pd


# resort the data by time and delete unused column
def process_files(source_dir, target_dir):
    # 获取源文件夹中所有的csv文件
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

    for file in csv_files:
        # 读取csv文件
        df = pd.read_csv(os.path.join(source_dir, file))

        # # 从filename列中提取数字
        # df['sequence'] = df['filename'].str.extract('-(\d+)\.jpg')[0]
        #
        # # 检查是否存在NaN值
        # if df['sequence'].isna().any():
        #     print(f"NaN value found in file: {file}")
        #     df['sequence'] = df['sequence'].fillna('some_value')  # 用适当的值替换'some_value'
        #
        # # 将sequence列转换为整数类型
        # df['sequence'] = df['sequence'].astype(int)
        # 从filename列中提取数字并转换为整数类型
        df['sequence'] = df['filename'].str.extract('-(\d+)\.jpg')[0].astype(int)

        # 根据序列号重新排序
        df = df.sort_values('sequence')

        # 删除filename、Unnamed: 0列和新添加的sequence列
        df = df.drop(['Unnamed: 0', 'filename', 'sequence'], axis=1)

        # 将处理后的数据保存到目标文件夹
        df.to_csv(os.path.join(target_dir, file), index=False)


# 调用函数，传入源文件夹路径和目标文件夹路径
source_dir = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial1\1"
target_dir = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\trial1_sorted\1"
process_files(source_dir, target_dir)
