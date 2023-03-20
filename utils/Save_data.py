import numpy as np
import torch
import os
from torch.utils.data  import Dataset, DataLoader
def Save_data(private_dataset, test_dataset, device_names_Save):
    for div_num in range(len(private_dataset)):
        for cnt in range(len(private_dataset[div_num])):  # TensorDataset 类型
            tensor_pri_tuple = private_dataset[div_num][cnt]
            # print(tensor_tuple[0:1])
            for tensor_pri_feature, tensor_pri_label in tensor_pri_tuple:
                tensor_pri_feature = tensor_pri_feature.reshape(-1, 115)
                tensor_pri_label  = tensor_pri_label.reshape(-1, 1)
                # print(f"{tensor_feature},{tensor_label}")
                inputs = [tensor_pri_feature, tensor_pri_label]
                outputs = torch.cat(inputs, dim=1)  # 初步形成Tensor.shape = (Sample_num,115+1)
                # print(outputs)
                with open(f"./output_data_P/non-iid_{device_names_Save[div_num]}_cnt{cnt}.csv", "a+") as f:
                    np.savetxt(f, outputs, delimiter=",")
                    f.close()
    tensor_test_tuple = test_dataset
    for tensor_test_feature, tensor_test_label in tensor_test_tuple:
        tensor_test_feature = tensor_test_feature.reshape(-1, 115)
        tensor_test_label = tensor_test_label.reshape(-1, 1)
        # print(f"{tensor_test_feature},{tensor_test_label}")
        inputs_T = [tensor_test_feature, tensor_test_label]
        outputs_T = torch.cat(inputs_T, dim=1)
        with open (f"./output_data_T/non-iid_TestData.csv", "a+") as f_T:
            np.savetxt(f_T, outputs_T, delimiter=",")
            f_T.close()





