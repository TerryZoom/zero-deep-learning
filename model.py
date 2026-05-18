import torch
import torch.nn as nn 
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
    
if __name__ == '__main__':
    rui = Rui()
    input = torch.ones((64, 3, 32, 32))
    output = rui(input)
    print(output.shape)