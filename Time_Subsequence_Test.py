import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model.InceptionTime import InceptionTime
from ReadData import load_data_with_index, MyDataset

inceptiontime = InceptionTime()
state_path = "./State_dict/InceptionTime_State/Time_based_split/Random_motion_11.pth"
inceptiontime.load_state_dict(torch.load((state_path)))
inceptiontime.eval()
writer = SummaryWriter("./Logs_tensorboard/Time_based_split/Random_motion/participants_time_diff")
with torch.no_grad():
    for i in range(1, 12):
        index = [1760, 1760 + 160 * i]

        accuracy_list = []
        for _ in range(5):
            testdata = load_data_with_index(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\Time_based_split\Random_motion\Random_motion_val", index)
            test_dataset = MyDataset(testdata)
            batch_size = 16
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            test_dataset_len = len(test_dataset)
            # print("The length of test dataset is :{}".format(test_dataset_len))
            total_test_accuracy = 0
            for data in test_loader:
                positions, targets = data
                outputs = inceptiontime(positions)
                accuracy = (outputs.argmax(1) == targets).sum()
                total_test_accuracy = total_test_accuracy + accuracy
            # print("Total test accuracy: {}".format(total_test_accuracy/test_dataset_len))
            accuracy_list.append((total_test_accuracy/test_dataset_len))

        Mean_accuracy = sum(accuracy_list) / len(accuracy_list)
        # print("Mean test accuracy:{}".format(Mean_accuracy))
        print("Mean test accuracy for index {}: {}".format(index, Mean_accuracy))

        writer.add_scalar('Mean_Accuracy', Mean_accuracy, i)

    writer.close()