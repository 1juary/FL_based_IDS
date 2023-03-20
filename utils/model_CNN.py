import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, output_len):  # 构造函数
        super().__init__()

        self.conv1 = nn.Conv1d(23, 64, 3, padding=1)

        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)

        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.dropoutcv2 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, 3, padding=1)

        self.conv6 = nn.Conv1d(128, 128, 3, stride=1, padding=1)  # padding 输入的每一条边补充0的层数
        self.conv7 = nn.Conv1d(128, 128, 3, stride=2, padding=1)
        self.dropoutcv3 = nn.Dropout(p=0.3)
        self.conv8 = nn.Conv1d(128, 128, 3, stride=2, padding=1)
        self.dropoutcv4 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(256, 128)  # 128*2 由于stride=2将输入23*5-》128*2 进行缩放

        self.fc2 = nn.Linear(128, output_len)
        self.output_cnt = output_len

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropoutcv2(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)

        x = self.dropoutcv3(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.dropoutcv4(x)

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)

        x = self.fc2(x)
        if self.output_cnt == 1:
            x = x.squeeze(dim=-1)
        return x


def GetNbaIotModel(output_len):
    model = CNN(output_len)
    return model

