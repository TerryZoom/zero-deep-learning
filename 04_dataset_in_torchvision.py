import torchvision
data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='data\CIFAR10',
                                         train=True,
                                         transform=data_transform,
                                         download=False) # alter to True if first run the code
test_set = torchvision.datasets.CIFAR10('data\CIFAR10',
                                         train=False,
                                         transform=data_transform,
                                         download=False)

print(test_set[0])
print(test_set.classes) # all classes of cifar 10
img, target = test_set[0]
print(img) # the pic 0 of test data
print(target) # the class num of img 0 in test data
print(test_set.classes[target]) # the name of this class
img_pil = torchvision.transforms.ToPILImage()(img)
img_pil.show()

writer = SummaryWriter('logs_04')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)
writer.close()
# python -m tensorboard.main --logdir=logs_04