import torch
import torch.nn as nn


class CustomModel2(nn.Module):
    def __init__(self):
        super(CustomModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.avgpool3 = nn.AvgPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        #self.subpixel = nn.PixelShuffle(16)
        #self.conv_output = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        item = x
        print("DEBUT")

        '''
        out = self.avgpool1(self.relu1(self.conv1(x)))
        print(out.shape)
        out = self.avgpool2(self.relu2(self.conv2(out)))
        print(out.shape)
        out = self.avgpool3(self.relu3(self.conv3(out)))
        '''
        out = self.relu1(self.conv1(x))
        print(out.shape)
        out = self.relu2(self.conv2(out))
        print(out.shape)
        out = self.relu3(self.conv3(out))
        print(out.shape)
        residual2 = out
        out = self.relu4(self.conv4(out))
        print("Avant add")
        print(out.shape)
        out = torch.add(out, residual2)
        print("Après add et avant subpixel")
        print(out.shape)
        out = self.subpixel(out)
        print("Après subpixel")
        print(out.shape)
        out = self.conv_output(out)
        out = torch.add(out, item)

        return out