import torch

from torch.utils.data import DataLoader

from Model.InceptionTime import InceptionTime
from ReadData import load_data_with_index, MyDataset

inceptiontime = InceptionTime()
state_path = "./State_dict/Trial0_subsequence_test/Prosaccades_gap/Fold_3.pth"
inceptiontime.load_state_dict(torch.load((state_path)))

accuracy_list = []
for i in range(5):
    testdata = load_data_with_index(r"D:\MyFiles\UOB_Robotics22\Dissertation\data_info\SubSequence\Trial0\Prosaccades_gap", [0, 800])
    test_dataset = MyDataset(testdata)
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_len = len(test_dataset)
    print("The length of test dataset is :{}".format(test_dataset_len))
    total_test_accuracy = 0
    for data in test_loader:
        positions, targets = data
        outputs = inceptiontime(positions)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_test_accuracy = total_test_accuracy + accuracy

    print("Total test accuracy: {}".format(total_test_accuracy/test_dataset_len))
    accuracy_list.append((total_test_accuracy/test_dataset_len))

Mean_accuracy = sum(accuracy_list) / len(accuracy_list)
print("Mean test accuracy:{}".format(Mean_accuracy))
