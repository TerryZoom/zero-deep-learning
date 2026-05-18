import torch
import torch.nn as nn
import torchvision

vgg16_false1 = torchvision.models.vgg16(pretrained = False)
vgg16_true1 = torchvision.models.vgg16(pretrained = True)
vgg16_true2 = torchvision.models.vgg16(pretrained = True)
vgg16_false2 = torchvision.models.vgg16(pretrained = False)


train_data = torchvision.datasets.CIFAR10(
    root='data/CIFAR10',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

# Part1: fix
vgg16_true1.add_module('added_linear', nn.Linear(1000, 10))
print(vgg16_true1)
print('-' * 100)

vgg16_true2.classifier.add_module('added_linear', nn.Linear(1000, 10))
print(vgg16_true2)
print('-' * 100)

vgg16_false1.classifier[6] = nn.Linear(4096, 10) # (4096, 1000) -> (4096, 10)
print(vgg16_false1)
print('-' * 100)

# Part2: save
# method1: cons + para
# save
torch.save(vgg16_false2, 'models/vgg16_method1.pth')
# use
model1 = torch.load('models/vgg16_method1.pth')
print(model1)
print('-' * 100)

# method2: only para(format: dict)
# save
torch.save(vgg16_false2.state_dict(), 'models/vgg16_method2.pth')
# use
model2 = torchvision.models.vgg16(pretrained = False) # must design same cons first
model2.load_state_dict(
    torch.load('models/vgg16_method2.pth')
)
model2.eval()
print(model2)