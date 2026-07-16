import torch
from torch import nn
from torchinfo import summary

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_conv_1_1=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_conv_1_1:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            x = self.conv3(x)
        out += x # Residual connection
        
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block): # block is ResNetBlock in the practice below
        super(ResNet18, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.part2 = nn.Sequential(
            block(64, 64),
            block(64, 64)
        )
        self.part3 = nn.Sequential(
            block(64, 128, stride=2, use_conv_1_1=True),
            block(128, 128)
        )
        self.part4 = nn.Sequential(
            block(128, 256, stride=2, use_conv_1_1=True),
            block(256, 256)
        )
        self.part5 = nn.Sequential(
            block(256, 512, stride=2, use_conv_1_1=True),
            block(512, 512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        x = self.part5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(ResNetBlock)
    model.to(device)
    print(summary(model, (1, 1, 224, 224))) # (batch_size, channels, height, width)