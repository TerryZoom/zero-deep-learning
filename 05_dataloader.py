from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt # attension is .pyplot

batch_size = 8
test_data = torchvision.datasets.CIFAR10(root='data\CIFAR10',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=False)
test_loader1 = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True, # shuffle the order of pic in dataset
                         num_workers=0,
                         drop_last=False # keep the redundant pic
                         )
test_loader2 = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True, # shuffle the order of pic in dataset
                         num_workers=0,
                         drop_last=False # keep the redundant pic
                         ) # dataloader is one-time, so use two
test_loader3 = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True, # shuffle the order of pic in dataset
                         num_workers=0,
                         drop_last=False # keep the redundant pic
                         ) # dataloader is one-time, so use two

# Part1: dataset : test_data
img, target = test_data[0]
print(img.shape)
print(target)
print('-' * 100)

# Part2: dataloader : test_loader
for data in test_loader1:
    imgs, targets = data
print(imgs.shape, targets) # only see the final batch

# imgs, targets = next(iter(test_loader))
# print(imgs.shape)
# print(targets) # the label num of pics in one batch
print('-' * 100)

# Part3: see in plt
grid = torchvision.utils.make_grid(imgs, nrow=4) # 4 pics in a row
plt.imshow(grid.permute(1, 2, 0)) # matplotlib (H, W, C)
# PyTorch (C, H, W)

plt.axis('off') # dont show axis
plt.show()

# Part4: see in board
writer = SummaryWriter('logs_05')
step = 0
for data in test_loader2:
    imgs, targets = data
    writer.add_images('batch_pic', imgs, step)
    step += 1

# Part5: shuffle
for epoch in range(2):
    step = 0
    for data in test_loader3:
        imgs, targets = data
        writer.add_images(f'shuffle/epoch{epoch + 1}',
                          imgs,
                          step)
        step += 1
writer.close() # python -m tensorboard.main --logdir=logs_05

'''
在drop_last为false的情况下，dataloader图片的数量和dataset是一样的，
但是它的迭代对象是batch数据，每个batch包含很多图片，取决于batch_size

iter(loader)让loader变成next能够接受的迭代器格式
next作用是只取目标对象的第一个迭代对象

比如next(iter(dataset))取的就是单个样本，只不过一般直接访问：dataset[0]
next(iter(dataloader))取的就是第一个batch数据
''' 
# dataloader 是一次性的
# utf-8