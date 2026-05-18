import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs_09')

# Part1: normal
class Rui1(nn.Module):
    def __init__(self):
        super(Rui1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2) # h_o = [h_i * 2 * padding - (kernel-1)] / stride + 1
        self.maxpool1 = nn.MaxPool2d(2) # hw: 32 -> 16
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2) # if dont change hw, padding dont change
        self.maxpool2 = nn.MaxPool2d(2) # hw: 8
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2) # hw: 4
        self.flatten = nn.Flatten() # c * h * w = 64 * 4 * 4 = 1024
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

rui1 = Rui1()
print(rui1)
print('-' * 100)
input = torch.ones((64, 3, 32, 32))
output = rui1(input)
print(output.shape)
writer.add_graph(rui1, input)

# Part2: sequential
class Rui2(nn.Module):
    def __init__(self):
        super(Rui2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), # h_o = [h_i * 2 * padding - (kernel-1)] / stride + 1
            nn.MaxPool2d(2), # hw: 32 -> 16
            nn.Conv2d(32, 32, 5, padding=2), # if dont change hw, padding dont change
            nn.MaxPool2d(2), # hw: 8
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2), # hw: 4
            nn.Flatten(), # c * h * w = 64 * 4 * 4 = 1024
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
            )
    def forward(self, x):
        x = self.model(x)
        return x

rui2 = Rui2()
print(rui2)
writer.add_graph(rui2, input)
writer.close() # python -m tensorboard.main --logdir=logs_09