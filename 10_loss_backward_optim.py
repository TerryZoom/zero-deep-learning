import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

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

# Part1: normal loss
outputs = torch.tensor([1., 2, 3])
outputs = torch.reshape(outputs, (1, 1, 1, 3))
targets = torch.tensor([1., 2, 5])
targets = torch.reshape(targets, (1, 1, 1, 3))

loss1 = nn.L1Loss(reduction='mean')
loss2 = nn.L1Loss(reduction='sum')
result1 = loss1(outputs, targets)
result2 = loss2(outputs, targets)
print(result1, result2)
print('-' * 100)

# Part2: cross_loss in exp - no epoch
lr = 0.01
loss3 = nn.CrossEntropyLoss()
rui1 = Rui()
optim = torch.optim.SGD(rui1.parameters(),
                        lr=lr)
for data in test_loader:
    imgs, targets = data
    outputs = rui1(imgs)
    
    result_loss = loss3(outputs, targets)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
print('the last sample loss in this batch: ', result_loss.item())
# attention item(): tensor -> num
print('-' * 100)

# Part3: cross_loss in exp
rui2 = Rui()
optim = torch.optim.SGD(rui2.parameters(),
                        lr=lr)
epoch_num = 20
for epoch in range(epoch_num):
    result_loss = 0.0
    batches_num = len(test_loader)

    for data in test_loader:
        imgs, targets = data
        outputs = rui2(imgs)

        loss = loss3(outputs, targets)

        optim.zero_grad() # grad to 0
        loss.backward() # back
        optim.step() # optim the para.

        result_loss += loss.item()

    epoch_loss = result_loss / batches_num
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

'''
result_loss 是每个 batch 的的 loss,
在 part2 里，只选了循环结束的最后一个 batch, 输出 loss

loss_per_epoch: 每个 epoch 看一遍完整的 loader(多个 batch)
累加所有的 batch 损失, 下一个 epoch 清零重来
''' # utf-8