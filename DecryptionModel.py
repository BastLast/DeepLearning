import torch
import torch.nn as nn
from torchvision import transforms


class DecryptionModel(nn.Module):
    def __init__(self):
        super(DecryptionModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.PReLU()
        self.linear = nn.Linear(9216, 9216)
        # self.unFlatten = nn.UnFlatten()

    def forward(self, x):
        # print("x shape : {}".format(x.shape))
        x = x.permute(1, 0, 2, 3)
        # print("x shape : {}".format(x.shape))
        blue = x[0]
        green = x[1]
        red = x[2]
        mix = torch.add(blue,green)
        mix = torch.add(mix,red)
        # print("blue shape : {}".format(blue.shape))
        mix = self.flatten(mix)
        mix = self.linear(mix)
        mix = self.relu1(mix)
        # print("blue shape : {}".format(blue.shape))
        blue = self.flatten(blue)
        blue = self.linear(blue)
        blue = self.relu1(blue)
        # self.linear().requires_grad = False
        green = self.flatten(green)
        green = self.linear(green)
        green = self.relu1(green)

        red = self.flatten(red)
        red = self.linear(red)
        red = self.relu1(red)

        blue = torch.reshape(blue, [blue.shape[0], 96, 96])
        green = torch.reshape(green, [green.shape[0], 96, 96])
        red = torch.reshape(red, [red.shape[0], 96, 96])
        #  print("blue shape : {}".format(blue.shape))
        # print("x shape : {}".format(x.shape))
        x = torch.stack((blue, green, red))
        # print("x shape : {}".format(x.shape))
        x = x.permute(1, 0, 2, 3)
        # print("x shape : {}".format(x.shape))
        return x
