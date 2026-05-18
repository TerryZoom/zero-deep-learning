# pool default see stride = kernel_size

from torch import nn
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
        super(Rui, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     ceil_mode=False) # default false: shed the redundant pix
    def forward(self, input):
        output = self.maxpool1(input)
        return output

rui = Rui()
writer = SummaryWriter('logs_07')
step = 0

for data in test_loader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    outputs = rui(imgs) # pool
    writer.add_images('output', outputs, step)
    step += 1
writer.close() # python -m tensorboard.main --logdir=logs_07