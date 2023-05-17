import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import importlib
import utils.utils as utils
from propagation_ASM import propagation_ASM
from utils.pytorch_prototyping.pytorch_prototyping import Unet 
import time

# Full Framework
class HoloNet_single(nn.Module):
    """
    Class initialization parameters
    -------------------------------
    distance: propagation distance between the SLM and the target plane 
    wavelength: the wavelength of the laser 
    feature_size: the pixel pitch of the SLM 
    initial_phase:  a module that is used to predict the initial phase at the target plane 
    final_phase_only: the module that is used to encode the complex wavefield at SLM plane into a phase-only hologram
    proptype: chooses the propagation operator. Default ASM.
    linear_conv: if True, pads for linear conv for propagation. Default True
    """
    def __init__(self, distance=0.1, wavelength=520e-9, feature_size=6.4e-6,
                  initial_phase=None, final_phase_only=None, proptype='ASM', linear_conv=True):

        super(HoloNet_single, self).__init__()

        # submodules
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only

        # propagation parameters
        self.wavelength = wavelength
        self.feature_size = (feature_size
                              if hasattr(feature_size, '__len__')
                              else [feature_size] * 2)
        self.distance = -distance


        # objects to precompute
        self.precomped_H = None

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
        else:
            ValueError(f'Unsupported prop type {proptype}')

        self.linear_conv = linear_conv

        # set a device for initializing the precomputed objects
        try:
            self.dev = next(self.parameters()).device
        except StopIteration:  # no parameters
            self.dev = torch.device('cpu')

        s = torch.tensor(0.95, requires_grad=True)
        self.s = torch.nn.Parameter(s)
          

    def forward(self, target_amp):
        # compute some initial phase, convert to real+imag representation

        init_phase = self.initial_phase(target_amp)
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        # target_complex = torch.stack((real, imag), -1)
        target_complex = torch.complex(real, imag)

        # precompute the propagation kernel once it is not provided
        if self.precomped_H is None:
            self.precomped_H = self.prop(target_complex,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance,
                                         return_H=True,
                                         linear_conv=self.linear_conv)
            self.precomped_H = self.precomped_H.to(self.device).detach()
            self.precomped_H.requires_grad = False


        # Propagate the wavefield to the SLM plane 
        slm_field = self.prop(target_complex, self.feature_size,
                              self.wavelength, self.distance,
                              precomped_H=self.precomped_H,
                              linear_conv=self.linear_conv)

        # Transform it to amplitude-phase form

        amp, ang = utils.rect_to_polar(slm_field.real, slm_field.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)

        pred_phase = self.final_phase_only(slm_amp_phase)

        return pred_phase

                                      
    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)

        # try setting dev based on some parameter, default to cpu
        try:
            slf.device = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.device = device_arg
        return slf

class HoloNet_full(nn.Module):
    """
    Class initialization parameters
    -------------------------------
    distance: propagation distance between the SLM and the target plane 
    wavelength: the wavelength of the laser 
    feature_size: the pixel pitch of the SLM 
    initial_phase:  a module that is used to predict the initial phase at the target plane 
    final_phase_only: the module that is used to encode the complex wavefield at SLM plane into a phase-only hologram
    proptype: chooses the propagation operator. Default ASM.
    linear_conv: if True, pads for linear conv for propagation. Default True
    """
    def __init__(self, distance=0.1, wavelengths=(680e-9, 520e-9, 450e-9), feature_size=6.4e-6,
                  initial_phase=None, final_phase_only=None, proptype='ASM', linear_conv=True):

        super(HoloNet_full, self).__init__()

        # submodules
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only

        # propagation parameters
        self.wavelengths = wavelengths
        self.feature_size = (feature_size
                              if hasattr(feature_size, '__len__')
                              else [feature_size] * 2)
        self.distance = -distance
        self.color = [0,1,2]


        # objects to precompute
        self.precomped_H = None

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
        else:
            ValueError(f'Unsupported prop type {proptype}')

        self.linear_conv = linear_conv

        # set a device for initializing the precomputed objects
        try:
            self.dev = next(self.parameters()).device
        except StopIteration:  # no parameters
            self.dev = torch.device('cpu')

        s = torch.tensor(0.95, requires_grad=True)
        self.s = torch.nn.Parameter(s)


    def forward(self, target_amp):
        # compute some initial phase, convert to real+imag representation

        init_phase = self.initial_phase(target_amp)
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        # target_complex = torch.stack((real, imag), -1)
        target_complex = torch.complex(real, imag)

        # precompute the propagation kernel once it is not provided


        slm_field = []
        for wv, i in zip(self.wavelengths, self.color):

            slm_complex = target_complex[:, i, :, :].unsqueeze(1)

            # Propagate the wavefield to the SLM plane 
            slm_field_c = self.prop(slm_complex, self.feature_size,
                                    wv, self.distance,
                                    precomped_H=None,
                                    linear_conv=self.linear_conv)
            slm_field.append(slm_field_c)

        slm_field = torch.cat(slm_field, dim=1)
        # Transform it to amplitude-phase form

        amp, ang = utils.rect_to_polar(slm_field.real, slm_field.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)

        pred_phase = self.final_phase_only(slm_amp_phase)

        return pred_phase


    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)

        # try setting dev based on some parameter, default to cpu
        try:
            slf.device = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.device = device_arg
        return slf
    

class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=1, num_out=1):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(num_in, num_out, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class PhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=4, num_out=1):
        super(PhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, num_out, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase
    
class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = nn.Sequential(
                     OutConv(32, 3),
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
        return out_phase

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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