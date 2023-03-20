import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.process_data_utils import GetDataset, SplitPrivateOpen, DilSplitPrivate, SplitPrivate, GetAllFeatureLabel, ShuffleDataset
from utils.train_utils import GetDeviceClassCat, GetDeviceClientCnt, reshape_sample

class SSFL_IDS_Client():   # 构建客户端的训练模型classify、discriminate
    def __init__(self, idx, 
                classify_dataset:torch.utils.data.Dataset, classify_model:nn.Module,classify_model_out_len, 
                classify_lr:float,
                discri_model,discri_model_out_len,discri_lr):

        # classify_model
        self.classify_model = classify_model  # 模型 类
        self.classify_dataset = classify_dataset  # 对每一个客户类 进行操作
        self.class_cat = classify_model_out_len if classify_model_out_len > 1 else 2
        # 11为classify_model准备 2 为discriminate_model 准备
        self.each_class_cnt = [0] * self.class_cat  # [0,0,...,0]  11个0 每一个类的具体数量
        for _, label in self.classify_dataset:  # 为每个设备中的流量类型分类
            self.each_class_cnt[label.item()] += 1
        self.classify_lr = classify_lr
        self.c_idx = idx
        self.classify_opt = optim.Adam(self.classify_model.parameters(), lr=self.classify_lr)
        # discriminate_model
        self.discri_model = discri_model
        self.discri_lr = discri_lr
        self.discri_opt = optim.Adam(self.discri_model.parameters(), lr=self.discri_lr)
        self.discri_model_out_len = discri_model_out_len  # 1
        if discri_model_out_len == 1:
            self.discri_loss_func = nn.BCEWithLogitsLoss()  # 损失函数之一
            # 比简单地将 Sigmoid 层加上 BCELoss 损失更稳定，因为使用了 log-sun-exp 技巧获得数值稳定性
        else:
            self.discri_loss_func = nn.CrossEntropyLoss()

        self.classify_model_out_len = classify_model_out_len  # 11

        if classify_model_out_len == 1:
            self.hard_label_loss_func = nn.BCEWithLogitsLoss()
            self.feature, self.label = self.classify_dataset[:]
            self.label = self.label.double()
            self.classify_dataset = torch.utils.data.TensorDataset(self.feature,self.label)
        else:
            self.hard_label_loss_func = nn.CrossEntropyLoss()  # 一种损失函数
        self.soft_label_loss_func = SSFL_IDS_CELoss()

class SSFL_IDS_CELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_pro, target_tensor):
        pred_pro = F.log_softmax(pred_pro, dim=1)
        out = -1 * pred_pro * target_tensor
        return out.sum() / len(pred_pro)

class SSFL_IDS_Server():
    def __init__(self, model, model_out_len, clients, dist_lr):
        self.model = model
        self.clients = clients
        self.client_cnt = len(clients)
        self.model_out_len = model_out_len
        self.dist_lr = dist_lr
        self.dist_opt = optim.Adam(self.model.parameters(), lr=self.dist_lr)
        self.soft_label_loss_func = SSFL_IDS_CELoss()
        if model_out_len != 1:
            self.hard_label_loss_func = nn.CrossEntropyLoss()
        else:
            self.hard_label_loss_func = nn.BCEWithLogitsLoss()

def Create_SSFL_IDS_Client(client_idx, private_dataset ,classify_model,classify_model_out_len, lr, discri_model,discri_model_out_len,discri_lr):
    client = SSFL_IDS_Client(client_idx,private_dataset,classify_model,classify_model_out_len,lr,discri_model,discri_model_out_len,discri_lr)
    return client  # client 类

def Create_SSFL_IDS_Server(server_model,classify_model_out_len,clients,dist_lr):
    server = SSFL_IDS_Server(server_model,classify_model_out_len,clients,dist_lr)
    return server

def CreateDataset(configs, dataset_name = "NBaIoT"):
    if dataset_name == "NBaIoT":
        return create_NBaIoT(configs)  # 准备好 testdataset、opendataset、privatedataset 按照比例分配后的数据

def create_NBaIoT(configs):
    prefix = "data/nba_iot_1000/" # 前缀

    device_names = [
        "Danmini_Doorbell/" , "Ecobee_Thermostat/", "Philips_B120N10_Baby_Monitor/",
        "Provision_PT_737E_Security_Camera/", "Provision_PT_838_Security_Camera/", "SimpleHome_XCS7_1002_WHT_Security_Camera/",
        "SimpleHome_XCS7_1003_WHT_Security_Camera/","Ennio_Doorbell/", "Samsung_SNH_1011_N_Webcam/",
    ]
    attack_names = [
        "benign", "g_combo", "g_junk","g_scan", "g_tcp", "g_udp",    # 11个
        "m_ack","m_scan","m_syn","m_udp","m_udpplain"
    ]
    if configs["classify_model_out_len"] == 1:
        attack_names = ["benign", "attack"]

# 具体不同设备划分的种类，契合每个目录名
    all_device_train_feature  = None
    all_device_train_label = None

# private open test 数据集
    all_device_private_feature = []
    all_device_private_label = []

    all_device_open_feature = None
    all_device_open_label = None

    all_device_test_feature = None
    all_device_test_label = None

    device_cnt = len(device_names)  # 9个

    if configs["load_data_from_pickle"] == False:
        for d_idx in range(device_cnt):  # 设备index
            cur_device_class_cat = GetDeviceClassCat(device_names[d_idx], configs["classify_model_out_len"])
            # 获得当前设备类别 是数字 也是 模型输出类型，为每种类型的具体攻击，11种，6种，2种（attack/benign）
            train_filenames = []
            test_filenames = []
            for i in range(len(attack_names)): # 攻击类别数组
                if (i < cur_device_class_cat):
                    train_filename = prefix + device_names[d_idx] + attack_names[i] + "_train.csv"
                    test_filename = prefix + device_names[d_idx] + attack_names[i] + "_test.csv"
                    train_filenames.append(train_filename) # 只是文件名,append增加列表，会把列表当作一个元素
                    test_filenames.append(test_filename)
            train_feature, train_label = GetAllFeatureLabel(train_filenames, configs["label_start_idx"])
            # 是在同一个设备中，返回所有分开的特征和标签
            # configs["label_start_idx"] 115 每个attack中都有115列，即115种特征
            # 标签是从0开始的数字，暗合attacks列表中的顺序

            private_feature, private_label, open_feature, open_label = SplitPrivateOpen(train_feature,train_label, configs["private_percent"]
                                                                                        , configs["open_percent"], cur_device_class_cat, False)
            # 上面特征和标签返回都是单独的列表，在顺序上一一对应
            all_device_private_feature.append(private_feature)  # 循环外面的全局参数 ，追加列表，整个列表作为一个参数，即，一个设备一个
            all_device_private_label.append(private_label)  #
            # 使用不同的拼接方法，因为设定不同
            if all_device_open_feature is None:
                all_device_open_feature = open_feature
                all_device_open_label = open_label
            else:
                all_device_open_feature = np.concatenate((all_device_open_feature, open_feature), axis=0)
                # 一维数组拼接，横向的,输出也是一整个列表
                all_device_open_label = np.concatenate((all_device_open_label, open_label), axis=0)

            # train的数据集，每个设备只有一个
            if all_device_train_feature is None:
                all_device_train_feature = train_feature # 和all_device_private_feature(取0.9)不同
                all_device_train_label = train_label

            # 拼接同所有设备中的test数据，且结合成两个内部一一对应的列表（数组）
            test_feature, test_label = GetAllFeatureLabel(test_filenames, configs["label_start_idx"])
            if all_device_test_feature is None:
                all_device_test_feature = test_feature
                all_device_test_label = test_label
            else:
                all_device_test_feature = np.concatenate((all_device_test_feature, test_feature), axis=0)
                all_device_test_label = np.concatenate((all_device_test_label, test_label), axis=0)

    scaler = MinMaxScaler()  # 最大最小归一化
    scaler.fit(all_device_open_feature)
    # open data 打乱
    all_device_open_feature = scaler.transform(all_device_open_feature)
    all_device_open_feature = reshape_sample(all_device_open_feature)  # (-1, 23, 5) （7120，23，25） 7120/80=89
    open_dataset = GetDataset(all_device_open_feature, all_device_open_label)  # tensor 张量类型
    # 形状打印
    # print(open_dataset)
    open_dataset = ShuffleDataset(open_dataset)
    # test data 打乱
    all_device_test_feature = scaler.transform(all_device_test_feature)
    all_device_test_feature = reshape_sample(all_device_test_feature)
    test_dataset = GetDataset(all_device_test_feature,all_device_test_label)  # 17800/200=89

    private_datasets = []  # 空列表
    for d_idx in range(device_cnt):  # 设备数量 9个
        cur_device_class_cat = GetDeviceClassCat(device_names[d_idx], configs["classify_model_out_len"])
        # 模型输出类型，为每种类型的具体攻击，11种，6种，2种（attack/benign）
        cur_device_client_cnt = GetDeviceClientCnt(device_names[d_idx], configs["device_client_cnt"],
                                                   configs["classify_model_out_len"])

        cur_device_private_feature = all_device_private_feature[d_idx]  # 列表元素是每一个设备的private data，也是列表
        cur_device_private_label = all_device_private_label[d_idx]

        cur_device_private_feature = scaler.transform(cur_device_private_feature)
        cur_device_private_feature = reshape_sample(cur_device_private_feature)
        # (-1, 23, 5)  # 例，有11个类别的设备中（7920.23，5）7920/（800*0.9）=11

        # private 数据有三种应用场景，见论文

        if configs["iid"] == True:
            cur_device_private_datasets = SplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                       cur_device_client_cnt, cur_device_class_cat, configs["iid"],
                                                       configs["data_average"])
            private_datasets.append(cur_device_private_datasets)
        elif configs["split"] == "dile":  # 迪利克雷分布，要分发的用户数量等于攻击种类
            # 迪利克雷分布，为假设不同client间的数据集  不满足独立同分布(Non-IID)
            cur_device_private_datasets = DilSplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                          cur_device_client_cnt, cur_device_class_cat,
                                                          configs["alpha_of_dile"], configs["seed"])
            private_datasets.append(cur_device_private_datasets)
        elif configs["split"] == "equally":
            cur_device_private_datasets = SplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                       cur_device_client_cnt, cur_device_class_cat, configs["iid"],
                                                       configs["data_average"])
            private_datasets.append(cur_device_private_datasets)

    return test_dataset, private_datasets, open_dataset  # 论文里了介绍的，处理数据集步骤