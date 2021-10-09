import torch
import torch.nn as nn
from torchvision import transforms


class DecryptionModel(nn.Module):
    def __init__(self, mille, centmille):
        super(DecryptionModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(9216, 9216)
        self.linear = nn.Linear(9216, 9216)
        self.linear = nn.Linear(9216, 9216)
        self.mille = mille
        self.centmille = centmille
        # self.unFlatten = nn.UnFlatten()

    def forward(self, x):
        print("x shape : {}".format(x.shape))
        x = x.permute(1, 0, 2, 3)
        #  print("x shape : {}".format(x.shape))
        blue = x[0]
        green = x[1]
        red = x[2]
        print("mille shape : {}".format(self.mille.shape))
        blue = torch.add(blue, self.mille)
        print("blue shape : {}".format(blue.shape))
        x = torch.add(green, blue)
        print("x shape : {}".format(x.shape))
        print("centmille : {}".format(self.centmille))
        print("centmille shape : {}".format(self.centmille.shape))
        print("red shape : {}".format(red.shape))
        red = torch.add(red, self.centmille)
        x = torch.add(x, red)
        print("red shape : {}".format(red.shape))
        print("x shape : {}".format(x.shape))
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu1(x)
        x = torch.reshape(x, [x.shape[0], 96, 96])
        violet = torch.sub(x, red)
        newblue = torch.sub(violet, green)
        newgreen = torch.sub(violet, blue)
        newblue = torch.sub(newblue, self.mille)
        newred = torch.sub(x, violet)
        newred = torch.sub(newred, self.centmille)
        x = torch.stack((newblue, newgreen, newred))
        #  print("x shape : {}".format(x.shape))
        x = x.permute(1, 0, 2, 3)
        # print("x shape : {}".format(x.shape))
        return x
