import torch
import torch.nn as nn 
import torchvision
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Rui(nn.Module):
    def __init__(self):
        super(Rui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2), # h_o = [h_i * 2 * padding - (kernel-1)] / stride + 1
            nn.MaxPool2d(2), # hw: 32 -> 16
            nn.Conv2d(32, 32, 5, 1, 2), # if dont change hw, padding dont change
            nn.MaxPool2d(2), # hw: 8
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2), # hw: 4
            nn.Flatten(), # c * h * w = 64 * 4 * 4 = 1024
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
            )
    def forward(self, x):
        x = self.model(x)
        return x

image_pth = r'data\naiwa_kongfu.jpg'
img = Image.open(image_pth)
print(img)
img_3c = img.convert('RGB')
print(img_3c)
img_3c.show()
print('-' * 100)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()]
     )

img_input = transform(img_3c)
img_input = img_input.to(device) # put to GPU
print(img_input)

model = torch.load(r'models\rui_epoch30_cuda.pth', 
                   map_location=torch.device(device))
model.to(device) # put to GPU
model.eval()
print(model)
print('-' * 100)

img_input = torch.reshape(img_input, (1, 3, 32, 32))
with torch.no_grad():
    output = model(img_input)
print(output)
print(output.argmax(1))