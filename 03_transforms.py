from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Part1 totensor
img_path = r'data\hymenoptera_data\train\bees\39672681_1302d204d1.jpg'
tensor_img = transforms.ToTensor()(Image.open(img_path))
print(tensor_img)
print('*' * 100)

# Part2 see in board
writer = SummaryWriter('logs_03')
writer.add_image('tensor_img', tensor_img)
writer.close()
# python -m tensorboard.main --logdir=logs_03

# Part3 review class usage, figure out  __init__/__call__/hi
class Person:
    def __init__(self, name):
        self.name = name
    def __call__(self, name):
        print('__call__ hi ' + name)
    def hi(self, name):
        print('hello ' + name)

# self is the sample -> rui
rui = Person('rui') # <- auto use __init__
rui('rui') # let sample be used in paren. = rui.__call__('rui')
rui.hi('rui') # also can be seen as Person.hi(rui, 'rui')

# Part4 norm
trans_norm = transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                  std = [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image('Norm', img_norm)
writer.close()
# python -m tensorboard.main --logdir=logs_03

# Part5 resize
trans_resize1 = transforms.Resize((512, 512))
img_resize1 = trans_resize1(Image.open(img_path))
img_resize1 = transforms.ToTensor()(img_resize1)

writer.add_image('resize', img_resize1, 0)
print(img_resize1)

trans_resize2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize2, 
                                    transforms.ToTensor()])
img_resize2 = trans_compose(Image.open(img_path))

writer.add_image('resize', img_resize2, 1)
print(img_resize2)
writer.close()

# Part6 random
trans_random = transforms.RandomCrop(512)
trans_compose_ran = transforms.Compose([transforms.Resize(512),
                                        trans_random,
                                        transforms.ToTensor()])
for i in range(10):
    img_crop = trans_compose_ran(Image.open(img_path))
    writer.add_image('random', img_crop, i)
writer.close() # python -m tensorboard.main --logdir=logs_03