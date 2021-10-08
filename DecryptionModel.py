import torch
import torch.nn as nn
from torchvision import transforms


class DecryptionModel(nn.Module):
    def __init__(self):
        super(DecryptionModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(9216, 9216)
        # self.unFlatten = nn.UnFlatten()

    def forward(self, x):
        print("x shape : {}".format(x.shape))
        x = self.conv1(x)
        # print("x shape : {}".format(x.shape))
        # x = self.flatten(x)
        # print("x shape : {}".format(x.shape))
        # x = self.linear(x)
        # print("x shape : {}".format(x.shape))
        x = self.relu1(x)
        # print("x shape : {}".format(x.shape))
        # x = torch.reshape(x, [x.shape[0], 1, 96, 96])
        print("x shape : {}".format(x.shape))
        x = self.conv2(x)
        print("x shape : {}".format(x.shape))

        return x
