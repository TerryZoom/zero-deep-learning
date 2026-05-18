import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

from model import * # model.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Part1: data
batch_size = 16
train_data = torchvision.datasets.CIFAR10(
    root='data/CIFAR10',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
test_data = torchvision.datasets.CIFAR10(
    root='data/CIFAR10',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False, # test loader dont shuffle
    num_workers=0,
    drop_last=False
)

# Part2: use
rui = Rui()
rui = rui.to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.SGD(rui.parameters(),
                            lr = lr)

train_step = 0
test_step = 0
epoch_num = 30
for epoch in range(epoch_num):
    print('-' * 10, f'第{epoch + 1}轮训练', '-' * 10)
    rui.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        output = rui(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 1000 == 0:
            print(f'训练次数: {train_step}, Loss: {loss.item():.4f}')
            writer.add_scalar('train_loss', loss.item(), train_step)
    
    rui.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = rui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item() # attention .item()
            avg_test_loss = total_test_loss / len(test_loader)
            
            acc = (outputs.argmax(1) == targets).sum().item()
            total_acc += acc

    print(f'整体平均Loss: {avg_test_loss:.4f}')
    print(f'整体acc: {total_acc / len(test_data):.4f}')
    writer.add_scalar('test_loss', total_test_loss, test_step)
    writer.add_scalar('test_acc', total_acc / len(test_data), test_step)
    test_step += 1

    if (epoch + 1) % 10 == 0:
        torch.save(rui, f'models/rui_epoch{epoch + 1}_{device}.pth')
        print('已保存')
writer.close()