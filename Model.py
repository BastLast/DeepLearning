import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        """
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        """
        #self.subpixel = nn.PixelShuffle(2)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv6(out)

        out = torch.add(out, x)

        return out
