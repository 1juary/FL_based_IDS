import torch
import torch.nn as nn
import torch.nn.functional as F
class MSCNN(nn.Module):
    def __init__(self, output_len):
        super().__init__()
        # 第一层卷积层，使用多尺度卷积核
        self.conv1_1 = nn.Conv1d(23, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(23, 16, kernel_size=5, padding=2)
        self.conv1_3 = nn.Conv1d(23, 16, kernel_size=7, padding=3)
        # 第二层卷积层，同样使用多尺度卷积核，并加入池化层进行降维
        self.conv2_1 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(48, 64, kernel_size=5, padding=2)
        self.conv2_3 = nn.Conv1d(48, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(128*3, 128)
        self.fc2 = nn.Linear(128, output_len)
        self.output_cnt = output_len
    def forward(self, x):
        # 第一层卷积层
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x3 = F.relu(self.conv1_3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        # 第二层卷积层
        x1 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x))
        x3 = F.relu(self.conv2_3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        # 全连接层
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.output_cnt == 1:
            x = x.squeeze(dim=-1)
        return x

def GetNbaIotModel(output_len):
    model = MSCNN(output_len)
    return model

