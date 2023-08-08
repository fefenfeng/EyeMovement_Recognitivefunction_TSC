import os
import shutil

# split whole data folder into 2 trials

folder_path = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\eye_centre_dataset_inc_blinks\right_eye"
folder_0 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial0\0"
folder_1 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial1\0"
folder_2 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial0\1"
folder_3 = r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\right_eye\trial1\1"

# 将文件夹路径放入列表中
folders = [folder_0, folder_1, folder_2, folder_3]

# 计数器初始化
counts = [0, 0, 0, 0]

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 从文件名中获取编号，并剔除扩展名
        name_without_ext, ext = os.path.splitext(filename)
        num = int(name_without_ext.split('_')[-1])

        # 复制文件到相应的文件夹
        shutil.copy(os.path.join(folder_path, filename), folders[num])

        # 更新计数器
        counts[num] += 1

# 打印每种文件的数量
for i, count in enumerate(counts):
    print(f"文件类型_{i} 的数量是: {count}")


