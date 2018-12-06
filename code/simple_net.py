import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv1d(69, 256, kernel_size=7, padding=0, stride=1)
        self.fc1 = nn.Linear(256, 14)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
