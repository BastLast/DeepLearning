import torch
import torch.nn as nn
from torchvision import transforms


class DecryptionModel(nn.Module):
    def __init__(self):
        super(DecryptionModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.relu1 = nn.PReLU()
        self.linear = nn.Linear(9216, 9216)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)
        blue = x[0]
        green = x[1]
        red = x[2]
        mix = torch.add(blue,green)
        mix = torch.add(mix,red)
        mix = self.flatten(mix)
        mix = self.linear(mix)
        mix = self.relu1(mix)
        blue = self.flatten(blue)
        blue = self.linear(blue)
        blue = self.relu1(blue)
        green = self.flatten(green)
        green = self.linear(green)
        green = self.relu1(green)
        red = self.flatten(red)
        red = self.linear(red)
        red = self.relu1(red)
        blue = torch.reshape(blue, [blue.shape[0], 96, 96])
        green = torch.reshape(green, [green.shape[0], 96, 96])
        red = torch.reshape(red, [red.shape[0], 96, 96])
        x = torch.stack((blue, green, red))
        x = x.permute(1, 0, 2, 3)
        return x
