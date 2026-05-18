# ReLU non-linear
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

batch_size = 8

test_data = torchvision.datasets.CIFAR10(
    root='data/CIFAR10',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

class Rui(nn.Module):
    def __init__(self):
        super(Rui, self).__init__()
        self.relu = nn.ReLU()  # non-linear
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()  # flatten
        self.linear = nn.Linear(32 * 32 * 3, 32 * 32 * 3)  # linear
    
    def forward(self, input):
        x = self.flatten(input)          # [8, 3072] channel, flattened num
        output_linear = self.linear(x)   # [8, 3072] channel, lineared num
        output_linear = output_linear.view(-1, 3, 32, 32) # reshape for board see

        output_relu = self.relu(input)
        output_sigmoid = self.sigmoid(input)

        return output_linear, output_relu, output_sigmoid

rui = Rui()
writer = SummaryWriter('logs_08')

step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output_linear, output_relu, output_sigmoid = rui(imgs)

    writer.add_images('output_linear', output_linear, step)
    writer.add_images('output_relu', output_relu, step)
    writer.add_images('output_sigmoid', output_sigmoid, step)
    step += 1
writer.close() # python -m tensorboard.main --logdir=logs_08