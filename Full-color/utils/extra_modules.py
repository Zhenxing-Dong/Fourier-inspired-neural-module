import torch.nn as nn 
import torch 
import numpy as np 
from utils.pytorch_prototyping.pytorch_prototyping import * 
import utils.channel as channel

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        #norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        norm_kwargs = dict(affine=True)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            #self.interlayer_norm = instance.InstanceNorm2D_wrap
            self.interlayer_norm = nn.BatchNorm2d

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)


class Generator(nn.Module):
    def __init__(self, filters=[960, 480, 240, 120, 60], in_channels=16, activation='relu',
                 n_residual_blocks=8, sample_noise=False,
                 noise_dim=32):

        super(Generator, self).__init__()
        
        kernel_dim = 3
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(affine=True)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = len(filters) - 1

        
        self.interlayer_norm = nn.BatchNorm2d

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm(in_channels, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(in_channels, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(in_channels=filters[0], channel_norm=False, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        for i in range(0, len(filters)-1):
            cur_block = nn.Sequential(
                nn.ConvTranspose2d(filters[i], filters[i+1], kernel_dim, **cnn_kwargs),
                self.interlayer_norm(filters[i+1], **norm_kwargs),
                self.activation(),
            )
            setattr(self, 'upconv_block%d'%(i+1), cur_block)


        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 1, kernel_size=(7,7), stride=1),
            nn.Hardtanh(-math.pi, math.pi)
        )


    def forward(self, x):        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        for k in range(self.n_upsampling_layers):
            cur_layer = getattr(self, 'upconv_block%d'%(k+1))
            x = cur_layer(x)

        out = self.conv_block_out(x)

        return out



class MultiScale_Encoder(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 out_channels=None,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()
        self.per_layer_out_ch = per_layer_out_ch

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.down_conv0 = DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm)
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                cur_layer = [DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm)]
                cur_layer.append(ResidualBlock(in_channels=per_layer_out_ch[i + 1]))
                setattr(self, 'down_conv%d'%(i+1), nn.Sequential(*cur_layer))

            
            self.direct_down1 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=4,
                                  padding=0,
                                  stride=4,
                                  bias=True if norm is None else False))

            self.direct_down2 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=8,
                                  padding=0,
                                  stride=8,
                                  bias=True if norm is None else False),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))

            smooth_layer = nn.Sequential(
                                Conv2dSame(int(3 * per_layer_out_ch[-1]), per_layer_out_ch[-1], kernel_size=3, bias = None),
                                norm(per_layer_out_ch[-1]),
                                nn.LeakyReLU(0.2))

            setattr(self, 'fusion_layer', smooth_layer)
            out_resLayer = ResidualBlock(in_channels= int(per_layer_out_ch[-1]))
            setattr(self, 'out_res', out_resLayer)

        if  out_channels != None:
            out_layer = Conv2dSame(int(per_layer_out_ch[-1]), out_channels, kernel_size=3, bias = True)
            setattr(self, 'out_layer', out_layer)

    def forward(self, input):
        out_down1 = self.direct_down1(input)
        out_down2 = self.direct_down2(input)
        x = input 
        for k in range(len(self.per_layer_out_ch)):
            cur_layer = getattr(self, 'down_conv%d'%(k))
            x = cur_layer(x)
        cat_features = torch.cat([out_down1, out_down2, x], dim=1)
        out = self.fusion_layer(cat_features)
        out = self.out_res(out)
        out = self.out_layer(out)
        return out


class MultiScale_Encoder(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 out_channels=None,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()
        self.per_layer_out_ch = per_layer_out_ch

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.down_conv0 = DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm)
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                cur_layer = [DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm)]
                cur_layer.append(ResidualBlock(in_channels=per_layer_out_ch[i + 1]))
                setattr(self, 'down_conv%d'%(i+1), nn.Sequential(*cur_layer))

            
            self.direct_down1 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=4,
                                  padding=0,
                                  stride=4,
                                  bias=True if norm is None else False))

            self.direct_down2 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=8,
                                  padding=0,
                                  stride=8,
                                  bias=True if norm is None else False),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))

            smooth_layer = nn.Sequential(
                                Conv2dSame(int(3 * per_layer_out_ch[-1]), per_layer_out_ch[-1], kernel_size=3, bias = None),
                                norm(per_layer_out_ch[-1]),
                                nn.LeakyReLU(0.2))

            setattr(self, 'fusion_layer', smooth_layer)
            out_resLayer = ResidualBlock(in_channels= int(per_layer_out_ch[-1]))
            setattr(self, 'out_res', out_resLayer)

        if  out_channels != None:
            out_layer = Conv2dSame(int(per_layer_out_ch[-1]), out_channels, kernel_size=3, bias = True)
            setattr(self, 'out_layer', out_layer)

    def forward(self, input):
        out_down1 = self.direct_down1(input)
        out_down2 = self.direct_down2(input)
        x = input 
        for k in range(len(self.per_layer_out_ch)):
            cur_layer = getattr(self, 'down_conv%d'%(k))
            x = cur_layer(x)
        cat_features = torch.cat([out_down1, out_down2, x], dim=1)
        out = self.fusion_layer(cat_features)
        out = self.out_res(out)
        out = self.out_layer(out)
        return out
