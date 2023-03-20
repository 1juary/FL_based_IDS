import torch
import numpy as np
from utils.process_data_utils import ShuffleDataset

def TrainWithFeatureLabel(dev,feature,label,batchsize,model,opt,loss_func):
    dataset = torch.utils.data.TensorDataset(feature, label)
    avg_loss = TrainWithDataset(dev,dataset,batchsize,model,opt,loss_func)
    return avg_loss

def TrainWithDataset(dev,dataset,batchsize,model,opt,loss_func):
    data_loader = torch.utils.data.DataLoader(  # dataloader 函数
                    dataset,
                    batch_size=batchsize,  # 每个batch加载多少个样本
                    shuffle=True,  # 设置为True时会在每个epoch重新打乱数据(默认: False)
                )
    model = model.to(dev)  # 将模型加载到指定设备上
    model.train()
    total_loss = 0
    for batch_idx, (batch_feature, batch_label) in enumerate(data_loader):
        # 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        opt.zero_grad()  # 先将梯度归零
        batch_feature = batch_feature.to(dev)
        batch_label = batch_label.to(dev)
        preds = model(batch_feature)  # 预测值，判断
        loss = loss_func(preds, batch_label)
        loss.backward()  # 反向传播计算得到每个参数的梯度值
        opt.step()  # 通过梯度下降执行一步参数更新
        total_loss += loss.item() * batch_label.size(0)
        #  item（）取出单元素张量的元素值并返回该值，保持该元素类型不变
    avg_loss = total_loss / len(dataset)
    return avg_loss

def EvalWithFeatureLabel(dev,feature,label,batchsize,model,loss_func):
    dataset = torch.utils.data.TensorDataset(feature, label)
    avg_loss = EvalWithDataset(dev,dataset,batchsize,model,loss_func)
    return avg_loss


def EvalWithDataset(dev,dataset,batchsize,model,loss_func):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(test_loader):
            feature = feature.to(dev)
            label = label.to(dev)
            out = model(feature)
            loss = loss_func(out, label)
            total_loss += (loss.item() * label.size(0))
    test_loss = total_loss / (len(dataset))
    return test_loss

def Predict(dev,feature,model,model_out_len):
    with torch.no_grad():
        model = model.to(dev)
        feature = feature.to(dev)
        logits = model(feature)  # 就是定义的神经网络的一层输出结果。该输出一般会再接一个softmax layer输出normalize 后的概率，用于多分类
        pred_label = Logits2PredLabel(logits, model_out_len)
        return pred_label

def Logits2PredLabel(logits,model_out_len):
    "pred -> hard label"
    with torch.no_grad():
        if model_out_len == 1:
            prediction = torch.round(torch.sigmoid(logits))  # 返回一个新张量，将输入input张量的每个元素舍入到最近的整数

        else:
            _, prediction = torch.max(logits, 1)
    return prediction

def Predict2SoftLabel(dev,feature,model,model_out_len):
    with torch.no_grad():
        model = model.to(dev)
        feature = feature.to(dev)
        logits = model(feature)
        logits = Logits2Soft(logits,model_out_len)
        return logits

def Logits2Soft(logits,model_out_len):
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(1)
    with torch.no_grad():
        if model_out_len == 1:
            logits = sigmoid(logits)
            soft_max_logits = torch.zeros(len(logits),2)
            for i in range(len(logits)):
                soft_max_logits[i] = torch.tensor([1-logits[i].item(),logits[i].item()])
            logits = soft_max_logits
        else:
            logits = softmax(logits)
        return logits

def Metrics(true_label,pred_label):
    # 评价矩阵还有待修改 Recall Accuracy F1-Score
    print("Evaluation...")
    all_prediction = pred_label.cpu()
    all_label = true_label.cpu()
    correct_num = (all_label == all_prediction).sum().item()  # 正确预测的标签数量 第一轮正确的仅仅只有1800个（共17800）
    test_acc = correct_num / (len(true_label))  # 以第一轮为例子 1800/17800 = 10.12%
    return correct_num, test_acc

def PredictWithDisUnknown(dev, open_feature, 
                        classify_model,classify_model_len_out_tensor,
                        discri_model, discri_model_len_out_tensor,
                        class_cat):

    discri_model = discri_model.to(dev)
    classify_model = classify_model.to(dev)

    average_tensor = [1.0 / class_cat ] * class_cat
    average_tensor = torch.tensor(average_tensor)

    with torch.no_grad():
        open_feature = open_feature.to(dev)  # discriminate_model 要用到open_data来训练

        wait_to_dis_label = classify_model(open_feature)
        wait_to_dis_label = Logits2Soft(wait_to_dis_label, classify_model_len_out_tensor)
        # classify model
        dis_label = discri_model(open_feature)  # 就是定义的神经网络的一层输出结果。该输出一般会再接一个softmax layer输出normalize 后的概率，用于多分类
        dis_label = Logits2PredLabel(dis_label, discri_model_len_out_tensor)
        # "pred -> hard label"
        # discriminate model
        for i in range(len(dis_label)):  # i 是标签的下标
            if dis_label[i].item() == 1:    # 被判断为unfamiliar，由于11种最大下标为10，所以11即为unfamiliar
                wait_to_dis_label[i] = average_tensor.clone()
        return wait_to_dis_label


def PredictAvg(dev,dataset,bounds,model):
    print()
    print("pred avg")
    print(bounds)
    each_label_avg_logit = {}
    soft_max = torch.nn.Softmax(1)
    model = model.to(dev)
    for i in range(len(bounds)):
        cur_class_start_idx = bounds[i][0]
        cur_class_end_idx = bounds[i][1]
        cur_label = i
        if cur_class_start_idx == cur_class_end_idx:
            continue
        else:
            cur_class_feature, cur_class_label = dataset[cur_class_start_idx:cur_class_end_idx]
            with torch.no_grad():
                cur_class_feature = cur_class_feature.to(dev)
                pred_label = model(cur_class_feature)
                pred_label = soft_max(pred_label)
                pred_label = torch.mean(pred_label, dim=0)
            cur_label = cur_class_label[0].item()
            each_label_avg_logit[cur_label] = pred_label.detach().clone()
    print("end pred avg")
    return each_label_avg_logit

def PredictFilter(dev, open_feature, classify_model,classify_model_len_out_tensor, class_cat, theta):
    print()
    print("in predict filter")
    print("theta = {}".format(theta))
    classify_model = classify_model.to(dev)
    average_tensor = [1.0 / class_cat ] * class_cat
    average_tensor = torch.tensor(average_tensor)

    with torch.no_grad():
        open_feature = open_feature.to(dev)
        wait_to_dis_label = classify_model(open_feature)
        wait_to_dis_label = Logits2Soft(wait_to_dis_label,classify_model_len_out_tensor)
        if theta < 0:
            max_num, pred_label = torch.max(wait_to_dis_label,1)
            theta = max_num.median()
        for i in range(len(wait_to_dis_label)):
            pred_label, pred_pro = torch.argmax(wait_to_dis_label[i]), torch.max(wait_to_dis_label[i])
            if pred_pro < theta:
                wait_to_dis_label[i] = average_tensor.clone()
        return wait_to_dis_label

def HardLabel(soft_label):
    sample_cnt = len(soft_label)  # 7120
    class_cat = len(soft_label[0])  # 11
    boundary = 1 / class_cat
    hard_label = [0] * sample_cnt
    for i in range(sample_cnt):
        cur_soft_label = soft_label[i]  # [0.0909,里面是dis_model判断出来的概率,0.0909]11个
        pred_label, pred_pro = torch.argmax(cur_soft_label), torch.max(cur_soft_label)  # 最大值下表，以及最大的概率值
        # torch.argmax()返回指定维度最大值的序号
        hard_label[i] = pred_label.item() if pred_pro > boundary else class_cat
        # hard_label 有7120（opendata）个数,如果预测值大于边界值 1/11
    return hard_label


def HardLabelVoteHard(all_client_hard_label, class_cat):
    client_cnt = len(all_client_hard_label)
    sample_cnt = len(all_client_hard_label[0])  # 第一种类型的样本数
    pred_labels = []
    label_votes_none_cnt = 0
    for i in range(sample_cnt):
        label_votes = [0] * class_cat
        for j in range(client_cnt):  # 每一位用户为某一种类型的流量投票，投7120轮
            cur_client_cur_sample_hard_label = all_client_hard_label[j][i]
            pred_label = cur_client_cur_sample_hard_label
            if pred_label != class_cat:  # 不是11就为某种类别的流量投票，11为默认的不清楚时候的值，若为11则返回0
                label_votes[pred_label] += 1
        if (len(label_votes) == 0):
            label_votes_none_cnt += 1
        max_vote_nums = max(label_votes)
        max_vote_idx = label_votes.index(max_vote_nums)
        pred_labels.append(max_vote_idx)
    pred_labels = torch.tensor(pred_labels)
    return pred_labels

def HardLabelVoteOneHot(all_client_hard_label, class_cat):
    client_cnt = len(all_client_hard_label)
    sample_cnt = len(all_client_hard_label[0])
    pred_labels = []
    all_vote_tensor = []
    label_votes_none_cnt = 0
    for i in range(sample_cnt):
        label_votes = [0] * class_cat
        for j in range(client_cnt):
            cur_client_cur_sample_hard_label = all_client_hard_label[j][i]
            pred_label = cur_client_cur_sample_hard_label
            if pred_label != class_cat:
                label_votes[pred_label] += 1
        if (len(label_votes) == 0):
            label_votes_none_cnt += 1
        max_vote_nums = max(label_votes)
        max_vote_idx = label_votes.index(max_vote_nums)
        pred_labels.append(max_vote_idx)
    all_one_hot_tensor = []
    for i in range(len(pred_labels)):
        cur_label = [0.0] * class_cat
        cur_label[pred_labels[i]] = 1.0
        all_one_hot_tensor.append(cur_label)
    all_vote_tensor = torch.tensor(all_one_hot_tensor)
    print("len of pred = {}".format(len(pred_labels)))
    print()
    return all_vote_tensor

def OneHot2Label(one_hot_vectors):
    _, labels = torch.max(one_hot_vectors, 1)
    labels = labels.double()
    return labels

def GetDeviceClientCnt(device_name, client_cnt, classify_model_out_len):
    if classify_model_out_len == 1:  # attack的选择只有[benign,attack]
        return 4
    else:
        if ((device_name == "Ennio_Doorbell/" or device_name == "Samsung_SNH_1011_N_Webcam/")):
            return int(client_cnt / 2) + 1  # 6

        else:
            return client_cnt  # 11

def GetDeviceClassCat(device_name,classify_model_out_len):  #
    if classify_model_out_len == 1:
        return 2  # 两类指 attack 和 benign
    if ((device_name == "Ennio_Doorbell/" or device_name == "Samsung_SNH_1011_N_Webcam/")):  # 只有六种分类的
        return 6
    else :
        return 11

def reshape_sample(feature):
    feature = np.reshape(feature, (-1, 23, 5))
    return feature

def PredUnknown(dev, feature, model, theta, model_out_len):
    # feature 为dis_train_feature，为discriminate_model的feature
    sure_unknown = []
    wait_to_dis = []
    soft_max = torch.nn.Softmax(1)  # Softmax可作为神经网络中的输出层，用于多分类，dim = 1指第二个维度
    # 将Softmax函数应用于一个n维输入张量，对其进行缩放，使n维输出张量的元素位于[0,1]范围内，总和为1
    sigmoid = torch.nn.Sigmoid()  # discriminate 二分类Sigmoid（）函数
    model = model.to(dev)
    feature = feature.to(dev)
    with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        # 在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
        # requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存
        out = model(feature)
        if model_out_len == 1:
            out = sigmoid(out)  # sigmoid 多用于二分类，在这里也就是discriminate
            soft_max_out = torch.zeros(len(out), 2)
            for i in range(len(out)):  # 7120
                soft_max_out[i] = torch.tensor([1-out[i].item(), out[i].item()])  # softmax 函数
            out = soft_max_out
        else:
            out = soft_max(out)
        max_num, pred_label = torch.max(out, 1)  # 返回输入张量所有元素的最大值  dim (int) – 指定的维度
        if theta < 0 :
            theta = max_num.median()  # 根据最大最小中位数 得到的某一个区间参数，是预测值
        for i in range(len(max_num)):
            if max_num[i] < theta:  # 置信函数，函数中的参数theta是作为一个判断条件，真正的theta要用max_num.median()计算
                sure_unknown.append(feature[i])  # 被认为是unfamiliar
            else:
                wait_to_dis.append(feature[i])
    if len(sure_unknown) == 0:
        return None 
    sure_unknown = torch.stack(sure_unknown)  # sure_unknown = 3558 wait_to_dis = 3562
    # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状

    return sure_unknown

def LabelFeature(feature,label):
    labels = [label] * len(feature)
    labels = torch.tensor(labels)
    return feature, labels

def DisUnknown(dev, client, dis_rounds, batchsize, dis_train_feature,theta):
    dis_train_feature = dis_train_feature.detach().clone()
    # detach()函数返回一个和源张量同shape、dtype和device的张量，并且与源张量共享数据内存，但不提供梯度的回溯
    sure_unknown_feature = PredUnknown(dev, dis_train_feature, client.classify_model, theta,
                                       client.classify_model_out_len)
    if sure_unknown_feature is None:  # opendata 之中没有判断出来为unfamiliar的，在训练的过程中会舍弃
        return False

    unknown_label_num = -1
    known_label_num = -1
    if client.discri_model_out_len == 1:
        unknown_label_num = 1.0
        known_label_num = 0.0
    else:
        unknown_label_num = 1
        known_label_num = 0

    sure_unknown_feature, sure_unknown_label = LabelFeature(sure_unknown_feature, unknown_label_num)  # unfamiliar类是classify_model判断出来的

    sure_known_feature, _ = client.classify_dataset[:]  # familiar类是private_data
    sure_known_feature = sure_known_feature.detach().clone()
    sure_known_feature, sure_known_label = LabelFeature(sure_known_feature, known_label_num)

    sure_unknown_feature = sure_unknown_feature.to(dev)
    sure_known_feature = sure_known_feature.to(dev)

    dis_feature = torch.cat((sure_known_feature, sure_unknown_feature), 0)
    # 按维数0拼接（竖着拼）  这里是一个用户有的private_data+被判断为unfamiliar的opendata

    sure_known_label = sure_known_label.to(dev)
    sure_unknown_label = sure_unknown_label.to(dev)

    dis_label = torch.cat((sure_known_label,sure_unknown_label), 0)

    cpu_dev = torch.device("cpu")
    dis_feature = dis_feature.to(cpu_dev)
    dis_label = dis_label.to(cpu_dev)

    dis_dataset = torch.utils.data.TensorDataset(dis_feature, dis_label)  # dataset化
    dis_dataset = ShuffleDataset(dis_dataset)  # 打乱

    for r in range(dis_rounds):  # 3
        TrainWithDataset(dev, dis_dataset, batchsize, client.discri_model, client.discri_opt, client.discri_loss_func)

    return True

