import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
#########################################

class UNet_fft(nn.Module):
    def __init__(self, bilinear=False, dim=32):
        super(UNet_fft, self).__init__()
        self.bilinear = bilinear
        self.dim = dim
        self.inc = FirstConv(1, dim)
        self.down1 = Down(dim, dim*2)
        self.down2 = Down(dim*2, dim*4)
        self.down3 = Down(dim*4, dim*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim*8, dim*16 // factor)
        self.up1 = Up(dim*16, dim*8 // factor, bilinear)
        self.up2 = Up(dim*8, dim*4 // factor, bilinear)
        self.up3 = Up(dim*4, dim*2 // factor, bilinear)
        self.up4 = Up(dim*2, dim, bilinear)
        self.outc = nn.Sequential(
                     OutConv(dim, 1),
                     nn.Hardtanh(-math.pi, math.pi)
        )
     
    def forward(self, x):     
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out_phase = self.outc(x9)
        return (torch.ones(1), out_phase)



class FirstConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.main(x)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


        self.main_fft = nn.Sequential(

            nn.Conv2d(in_channels*2, mid_channels*2, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels*2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(mid_channels*2, out_channels*2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels*2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), dim=(-2, -1), norm="ortho")
        return self.main(x) + y





class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)






























































