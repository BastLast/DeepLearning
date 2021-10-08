import torch
import torch.nn as nn

# pixelshuffle https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
# prelu instead of relu https://medium.com/@shoray.goel/prelu-activation-e294bb21fefa
# torch.add https://pytorch.org/docs/stable/generated/torch.add.html
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.residualA = _Residual_Block()
        self.residualB = _Residual_Block()
        self.residualC = _Residual_Block()

        self.recon = Recon_Block()
        # concat

        self.conv_mid = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        item = x
        out = self.relu1(self.conv_input(x))
        out = self.relu2(self.conv_down(out))

        out1 = self.residualA(out)
        out2 = self.residualB(out1)
        out3 = self.residualC(out2)

        recon1 = self.recon(out1)
        recon2 = self.recon(out2)
        recon3 = self.recon(out3)

        out = torch.cat([recon1, recon2, recon3], 1)
        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out = self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, item)

        return out