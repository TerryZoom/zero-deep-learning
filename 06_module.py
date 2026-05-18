import torch
import torch.nn as nn
from torch.nn import Conv2d
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

batch_size = 8
test_data = torchvision.datasets.CIFAR10(root='data\CIFAR10',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=False)
test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True, # shuffle the order of pic in dataset
                         num_workers=0,
                         drop_last=False # keep the redundant pic
                         )

class Rui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=3,
                            stride=1,
                            padding=0)
    def forward(self, x):
        x = self.conv1(x) # tensor -> conved tensor
        return x

rui = Rui()
writer = SummaryWriter('logs_06')

step = 0
for data in test_loader:
    imgs, targets = data
    outputs = rui(imgs)
    writer.add_images('inputs', imgs, step)

    # CIFAR10: 32×32 → conv(3×3, no padding) → 30×30
    outputs = torch.reshape(outputs,
                           (-1, 6, 30, 30))
    writer.add_images('outputs', outputs[:, :3, :, :], step) # only use 0-2 channel for tensorboard
    step += 1

print('last imgs: ', imgs.shape)
print('last output: ', outputs.shape)
writer.close() # python -m tensorboard.main --logdir=logs_06