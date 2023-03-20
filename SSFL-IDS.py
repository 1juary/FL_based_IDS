import sys
sys.path.append("utils/")
import torch
import numpy as np
from utils.model_utils import GetNbaIotModel
from utils.creator import CreateDataset, Create_SSFL_IDS_Client, Create_SSFL_IDS_Server
from utils.process_data_utils import GetFeatureFromOpenDataset
from utils.train_utils import TrainWithDataset, PredictWithDisUnknown, \
    TrainWithFeatureLabel, Predict, Metrics, DisUnknown, OneHot2Label, HardLabelVoteHard, HardLabel, GetDeviceClientCnt
from utils.Save_data import Save_data

def SSFL_IDS(conf, dev, clients, server, test_dataset, open_dataset):
    comm_cnt = conf["comm_cnt"]  # 201
    open_idx_set_cnt = conf["open_idx_set_cnt"]  # 10000
    batchsize = conf["batchsize"]  # 100
    train_rounds = conf["train_rounds"]  # 3
    dis_rounds = conf["discri_rounds"]  # 3
    dist_rounds = conf["dist_rounds"]  # 10
    theta = conf["theta"]  # -1
    labels = conf["labels"]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    first_train_rounds = conf["first_train_rounds"]  # 3
    class_cat = conf["classify_model_out_len"] if conf["classify_model_out_len"] > 1 else 2  # 11
    dis_train_cnt = 10000
    start_idx = 0
    end_idx = start_idx + open_idx_set_cnt  # 10000
    open_len = len(open_dataset)  # 7120

    for e in range(comm_cnt):  # 201
        sure_unknown_none = set()  # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
        all_client_hard_label = []
        open_feature, open_label = GetFeatureFromOpenDataset(open_dataset, start_idx, end_idx)
        # start_idx = 0 end_idx=10000  实质上是end_idx = 10000,超出全部的数据量7120,结合👇的if-else代码，意在取一定数量的open_data
        if open_idx_set_cnt > open_len:  # 10000 > 7120
            global_logits = torch.zeros(open_len, len(labels))  # 7120*11
        else:
            global_logits = torch.zeros(open_idx_set_cnt, len(labels))
        client_cnt = len(clients)  # 89
        participate = 0

        print("Round {} Stage I".format(e+1))

        for c_idx in range(client_cnt):  # 89

            print("Client {} Training...".format(c_idx+1))

            cur_client = clients[c_idx]  # 这里的clients是包含了数据和模型的
            cur_train_rounds = train_rounds if e != 0 else first_train_rounds

            if len(cur_client.classify_dataset) == 0:  # 如果某个设备的private_data被分配了0个数据就跳过
                continue

            for train_r in range(cur_train_rounds):
                TrainWithDataset(dev, cur_client.classify_dataset, batchsize, cur_client.classify_model,  # 训练classify model
                                              cur_client.classify_opt, cur_client.hard_label_loss_func)
                # 返回值是avg_loss，但这里没有定义返回值
            if sum(i > 0 for i in cur_client.each_class_cnt) == 1:  # 是每个客户当前每种类型的流量的个数，如果被认为只有一种流量就下一个客户
                continue  # 跳过当前循环的剩余语句，然后继续进行下一轮循环
            else:
                participate += 1
            dis_train_feature, _ = GetFeatureFromOpenDataset(open_dataset, 0, dis_train_cnt)
            #  feature, label = open_dataset[start_idx:end_idx]    返回的是open_data的feature 和 label
            # 一个客户的classify_model被训练好
            succ = DisUnknown(dev, cur_client, dis_rounds, batchsize, dis_train_feature, theta)
            # 这一步实际上是混合private和opendata中的unfamiliar，然后训练discriminate_model
            # discriminate model 判断出来认为是unfamiliar的数据，这里只输入open_data的feature和theta-1

            if succ == False:
                sure_unknown_none.add(c_idx)

            cur_client_open_feature = open_feature.detach().clone()

            if c_idx not in sure_unknown_none:  # 如果某一个用户 不在sure_unknown_none之中的话，即训练过discriminate_model
                local_logit = PredictWithDisUnknown(dev, cur_client_open_feature,  # 这里是所有的open_data
                                                    cur_client.classify_model, cur_client.classify_model_out_len,
                                                    cur_client.discri_model, cur_client.discri_model_out_len,
                                                    len(labels))
                # discriminate model的初始神经网络输出参数
                copy_local_logit = local_logit.detach().clone()

                hard_label = HardLabel(copy_local_logit)  # shape.size[7120,11]
                all_client_hard_label.append(hard_label)

        print()

        global_logits = HardLabelVoteHard(all_client_hard_label, class_cat)
        # 服务器投票

        print("Round {} Stage II".format(e+1))

        for c_idx in range(len(clients)):
            cur_client = clients[c_idx]
            print("Client {} Distillation Training...".format(c_idx+1))

            for r in range(dist_rounds):  # 10
                cur_global_logits = global_logits.detach().clone()  # 7120  [0,...,0]
                cur_client_open_feature = open_feature.detach().clone()
                if cur_client.classify_model_out_len != 1:  # 11
                    TrainWithFeatureLabel(dev, cur_client_open_feature, cur_global_logits, batchsize,  # 这里的cur_global_logits 实际上是预测的label
                                                       cur_client.classify_model, cur_client.classify_opt,
                                                       cur_client.hard_label_loss_func)
                else:
                    cur_global_logits = OneHot2Label(cur_global_logits)
                    TrainWithFeatureLabel(dev, cur_client_open_feature, cur_global_logits, batchsize,
                                                       cur_client.classify_model, cur_client.classify_opt,
                                                       cur_client.hard_label_loss_func)
        print()

        print("Server Training...")

        for dist_i in range(dist_rounds):
            cur_global_logits = global_logits.detach().clone()
            server_open_feature = open_feature.detach().clone()
            if server.model_out_len != 1:
                TrainWithFeatureLabel(dev, server_open_feature, cur_global_logits, batchsize,  #
                                                          server.model, server.dist_opt, server.hard_label_loss_func)
            else:
                cur_global_logits = OneHot2Label(cur_global_logits)
                TrainWithFeatureLabel(dev, server_open_feature, cur_global_logits, batchsize,
                                                          server.model, server.dist_opt, server.hard_label_loss_func)

        test_feature, test_label = test_dataset[:]
        pred_label = Predict(dev, test_feature, server.model, server.model_out_len)
        correct_num, test_acc = Metrics(test_label, pred_label)

        print("Round {} Test Acc = {} ".format(e+1, test_acc))
        print()

def SSFL_IDS_NBaIoT():
    configs = {
        "comm_cnt": 201,
        "device_client_cnt": 11,
        "private_percent": 0.9,
        "batchsize": 100,
        "iid": False,
        "need_dist": True,
        "open_percent": 0.1,
        "label_lr": 0.0001,
        "dist_lr": 0.0001,
        "discri_lr": 0.0001,
        "train_rounds": 3,
        "discri_rounds": 3,
        "dist_rounds": 10,
        "first_train_rounds": 3,
        "open_idx_set_cnt": 10000,
        "discri_cnt": 10000,
        "dist_T": 0.1,
        "need_SA": False,
        "test_batch_size": 256,
        "label_start_idx": 115,
        "test_round": 1,
        "data_average": True,
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "clien_need_dist_opt": False,
        "discri_model_out_len": 1,
        "classify_model_out_len": 11,
        "sample_cnt": 1000,
        "random": True,
        "vote": True,
        "seed": 7,
        "load_data_from_pickle": False,
        "soft_label": False,
        "num_after_float": 4,
        "theta": -1,
        "split": "dile",
        "alpha_of_dile": 0.1,
    }

    if configs["seed"] is not None: 
        np.random.seed(configs["seed"]) # 7
    
    device_names = [
        "Danmini_Doorbell/", "Ecobee_Thermostat/", "Philips_B120N10_Baby_Monitor/",
        "Provision_PT_737E_Security_Camera/", "Provision_PT_838_Security_Camera/", "SimpleHome_XCS7_1002_WHT_Security_Camera/",
        "SimpleHome_XCS7_1003_WHT_Security_Camera/", "Ennio_Doorbell/", "Samsung_SNH_1011_N_Webcam/",
    ]  # 每个设备的种类 先11个 后6

    device_names_Sava = [
        "Danmini_Doorbell", "Ecobee_Thermostat", "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera", "Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam",
    ]  # 为保存文件用，没有下划线，以防转义错误

    device_cnt = len(device_names) # 9

    clients = []
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 使用显卡计算
    client_idx = 0
    test_dataset, private_dataset, open_dataset = CreateDataset(configs, "NBaIoT")  # 0.2 0.7 0.1

    # 保存数据以供Fed其他模型使用
    # Save_data(private_dataset, test_dataset, device_names_Sava,)   # 最好追加，设备名称和用户编号

    for d_idx in range(device_cnt):  # 9
        cur_device_client_cnt = GetDeviceClientCnt(device_names[d_idx], configs["device_client_cnt"], configs["classify_model_out_len"])
        # 第一个设备11
        cur_device_private_datasets = private_dataset[d_idx]  # 当前设备的隐私数据
        for i in range(cur_device_client_cnt):
            classify_model_out_len = configs["classify_model_out_len"]  # 模型输出type种类
            classify_model = GetNbaIotModel(classify_model_out_len)  # 建立模型
            discri_model_out_len = configs["discri_model_out_len"]  # 1  判断是否familiar
            discri_model = GetNbaIotModel(discri_model_out_len)
            client = Create_SSFL_IDS_Client(client_idx, cur_device_private_datasets[i], classify_model, classify_model_out_len,
                                            configs["label_lr"], discri_model, discri_model_out_len, configs["discri_lr"])
            #
            clients.append(client)
            client_idx += 1

    server_model = GetNbaIotModel(configs["classify_model_out_len"])
    server = Create_SSFL_IDS_Server(server_model, configs["classify_model_out_len"], clients, configs["dist_lr"])  # Server 类
    SSFL_IDS(configs, dev, clients, server, test_dataset, open_dataset)  # clients是多元素的列表，server的是一个实例

if __name__ == "__main__":
    SSFL_IDS_NBaIoT()