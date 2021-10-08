import torch
import torch.nn as nn


class DecryptionModel(nn.Module):
    def __init__(self):
        super(DecryptionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(27648, 27648)
        # self.unFlatten = nn.UnFlatten()
       # self.reShape = torch.reshape(x,[96,96,3])

    def forward(self, x):
        x = self.flatten(x)
        print("x shape : {}".format(x.shape))
        x = self.linear(x)
        print("x shape : {}".format(x.shape))
        x = self.relu1(x)
        print("x shape : {}".format(x.shape))
       # x = self.reShape(x)
        print("x shape : {}".format(x.shape))
        return x
