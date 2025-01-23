import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMatchingNet(nn.Module):
    def __init__(self):
        super(FeatureMatchingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 2)  # 输出2个值表示x和y方向的位移

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x

class ReprojectionNet(nn.Module):
    def __init__(self):
        super(ReprojectionNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入3个值：x, y, z
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # 输出3个值：delta_x, delta_y, weight

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x