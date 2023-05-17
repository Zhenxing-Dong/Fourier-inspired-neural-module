import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.perceptualloss as perceptualloss

from propagation_model import ModelPropagate
from holonet import *
from unet_fft import *
# from deformable_unet import *
# from holoencoder import *
from utils.augmented_image_loader import ImageLoader
from warmup_scheduler import GradualWarmupScheduler

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--run_id', type=str, default='', help='Experiment name', required=True)
p.add_argument('--proptype', type=str, default='ASM', help='Ideal propagation model')
p.add_argument('--generator_path', type=str, default='', help='Torch save of Holonet, start from pre-trained gen.')
p.add_argument('--model_path', type=str, default='./models', help='Torch save CITL-calibrated model')
p.add_argument('--num_epochs', type=int, default=40, help='Number of epochs')
p.add_argument('--batch_size', type=int, default=10, help='Size of minibatch')
p.add_argument('--lr', type=float, default=1e-3, help='learning rate of Holonet weights')
p.add_argument('--scale_output', type=float, default=0.95,
               help='Scale of output applied to reconstructed intensity from SLM')
p.add_argument('--loss_fun', type=str, default='vgg-low', help='Options: mse, l1, si_mse, vgg, vgg-low')
p.add_argument('--purely_unet', type=utils.str2bool, default=True, help='Use U-Net for entire estimation if True')
p.add_argument('--model_lut', type=utils.str2bool, default=True, help='Activate the lut of model')
p.add_argument('--disable_loss_amp', type=utils.str2bool, default=True, help='Disable manual amplitude loss')
p.add_argument('--num_latent_codes', type=int, default=2, help='Number of latent codes of trained prop model.')
p.add_argument('--step_lr', type=utils.str2bool, default=False, help='Use of lr scheduler')
p.add_argument('--perfect_prop_model', type=utils.str2bool, default=False,
               help='Use ideal ASM as the loss function')
p.add_argument('--manual_aberr_corr', type=utils.str2bool, default=True,
               help='Divide source amplitude manually, (possibly apply inverse zernike of primal domain')

p.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')         
p.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup') 
p.add_argument('--kernel_length', type=int, default=2, help='Gaussian Filter kernel length')
p.add_argument('--sigma', type=float, default=0.5, help='Gaussian Filter sigma')  



# parse arguments
opt = p.parse_args()
channel = opt.channel
run_id = opt.run_id
loss_fun = opt.loss_fun
warm_start = opt.generator_path != ''
chan_str = ('red', 'green', 'blue')[channel]

# tensorboard setup and file naming
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
writer = SummaryWriter(f'runs/{run_id}_{chan_str}_{time_str}')


##############
# Parameters #
##############

# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# Propagation parameters
prop_dist = (7 * cm, 7 * cm, 7 * cm)[channel]
wavelength = (671 * nm, 543 * nm, 473 * nm)[channel]
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
homography_res = (1072, 1920)  # for CITL crop, see ImageLoader

# Training parameters
device = torch.device('cuda')
use_mse_init = False  # first 500 iters will be MSE regardless of loss_fun

# Image data for training
data_path = './data/div2k'  # path for training data

if opt.model_path == '':
    opt.model_path = f'./models/{chan_str}.pth'

# resolutions need to be divisible by powers of 2 for unet
if opt.purely_unet:
    image_res = (1072, 1920)  # 4 down layers
else:
    image_res = (1072, 1920)  # 4 down layers


###############
# Load models #
###############

# re-use parameters from CITL-calibrated model for out Holonet.
if opt.perfect_prop_model:
    final_phase_num_in = 2

    # set model instance as naive ASM
    model_prop = ModelPropagate(distance=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                                use_conv1d_mlp=False, num_latent_codes=[0],
                                norm=None, blur=None, content_field=False, proptype=opt.proptype).to(device)

    zernike_coeffs = None
    source_amplitude = None
    latent_codes = None
    u_t = None
else:
    if opt.manual_aberr_corr:
        final_phase_num_in = 2 + opt.num_latent_codes
    else:
        final_phase_num_in = 4
    blur = utils.make_kernel_gaussian(0.849, 3)

    # load camera model and set it into eval mode
    model_prop = ModelPropagate(distance=prop_dist,
                                feature_size=feature_size,
                                wavelength=wavelength,
                                blur=blur).to(device)
    model_prop.load_state_dict(torch.load(opt.model_path, map_location=device))

    # Here, we crop model parameters to match the Holonet resolution, which is slightly different from 1080p.
    # parameters for CITL model
    zernike_coeffs = model_prop.coeffs_fourier
    source_amplitude = model_prop.source_amp
    latent_codes = model_prop.latent_code
    latent_codes = utils.crop_image(latent_codes, target_shape=image_res, pytorch=True, stacked_complex=False)

    # content independent target field (See Section 5.1.1.)
    u_t_amp = utils.crop_image(model_prop.target_constant_amp, target_shape=image_res, stacked_complex=False)
    u_t_phase = utils.crop_image(model_prop.target_constant_phase, target_shape=image_res, stacked_complex=False)
    real, imag = utils.polar_to_rect(u_t_amp, u_t_phase)
    u_t = torch.complex(real, imag)

    # match the shape of forward model parameters (1072, 1920)

    # If you make it nn.Parameter, the Holonet will include these parameters in the torch.save files
    model_prop.latent_code = nn.Parameter(latent_codes.detach(), requires_grad=False)
    model_prop.target_constant_amp = nn.Parameter(u_t_amp.detach(), requires_grad=False)
    model_prop.target_constant_phase = nn.Parameter(u_t_phase.detach(), requires_grad=False)

    # But as these parameters are already in the CITL-calibrated models,
    # you can load these from those models avoiding duplicated saves.

model_prop.eval()  # ensure freezing propagation model

# create new phase generator object
if opt.purely_unet:
    phase_generator = UNet_fft().to(device)
    #   phase_generator = Holoencoder().to(device)
else:
    print('naive HoloNet ASM')
    phase_generator = HoloNet(
                            distance=prop_dist,
                            wavelength=wavelength,
                            feature_size=feature_size,
                            zernike_coeffs=zernike_coeffs,
                            source_amplitude=source_amplitude,
                            initial_phase=InitialPhaseUnet(4, 16),
                            final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2),
                            manual_aberr_corr=opt.manual_aberr_corr,
                            target_field=u_t,
                            latent_codes=latent_codes,
                            proptype=opt.proptype
                            ).to(device)
    print('HoloNet with fft ASM')
    # phase_generator = HoloNet(
    #                         distance=prop_dist,
    #                         wavelength=wavelength,
    #                         feature_size=feature_size,
    #                         zernike_coeffs=zernike_coeffs,
    #                         source_amplitude=source_amplitude,
    #                         initial_phase=InitialPhaseUnet_FFT(),
    #                         final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2),
    #                         manual_aberr_corr=opt.manual_aberr_corr,
    #                         target_field=u_t,
    #                         latent_codes=latent_codes,
    #                         proptype=opt.proptype
    #                         ).to(device)


if warm_start:
    phase_generator.load_state_dict(torch.load(opt.generator_path, map_location=device))

phase_generator.train()  # generator to be trained


###################
# Set up training #
###################

# loss function
loss_fun = ['mse', 'l1', 'si_mse', 'vgg', 'ssim', 'vgg-low', 'l1-vgg'].index(loss_fun.lower())

if loss_fun == 0:        # MSE loss
    print('use MSELoss')
    loss = nn.MSELoss()
elif loss_fun == 1:      # L1 loss
    loss = nn.L1Loss()
elif loss_fun == 2:      # scale invariant MSE loss
    loss = nn.MSELoss()
elif loss_fun == 3:      # vgg perceptual loss
    loss = perceptualloss.PerceptualLoss()
elif loss_fun == 5:
    loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
    loss_fun = 3

mse_loss = nn.MSELoss()

class TVLoss(torch.nn.Module):
    def __init__(self, tv_loss_weight=0.001):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

tv_loss = TVLoss()

# upload to GPU
loss = loss.to(device)
mse_loss = mse_loss.to(device)
tv_loss = tv_loss.to(device)

if loss_fun == 2:
    scaleLoss = torch.ones(1)
    scaleLoss = scaleLoss.to(device)
    scaleLoss.requires_grad = True

    optvars = [scaleLoss, *phase_generator.parameters()]
else:
    optvars = phase_generator.parameters()

# set aside the VGG loss for the first num_mse_epochs
if loss_fun == 3:
    vgg_loss = loss
    loss = mse_loss

# create optimizer
if warm_start:
    opt.lr /= 10

#   optimizer = optim.Adam(optvars, lr=opt.lr)
optimizer = optim.AdamW(optvars, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)

# learning rate scheduler

if opt.step_lr:
    scheduler = optim.lr_scheduler.StepLR(optimizer, 500, 0.5)
# warmup_epochs = opt.warmup_epochs
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.num_epochs-warmup_epochs, eta_min=1e-6)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)


# loads images from disk, set to augment with flipping
image_loader = ImageLoader(data_path,
                           channel=channel,
                           batch_size=opt.batch_size,
                           image_res=image_res,
                           homography_res=homography_res,
                           shuffle=True,
                           vertical_flips=True,
                           horizontal_flips=True)

num_mse_iters = 500
num_mse_epochs = 1 if use_mse_init else 0
opt.num_epochs += 1 if use_mse_init else 0

#################
# Training loop #
#################

for i in range(opt.num_epochs):

    epoch_loss = 0
    # switch to actual loss function from mse when desired
    if i == num_mse_epochs:
        if loss_fun == 3:
            loss = vgg_loss

    for k, target in enumerate(image_loader):
        # cap iters on the mse epoch(s)
        if i < num_mse_epochs and k == num_mse_iters:
            break

        # get target image
        target_amp, target_res, target_filename = target
        target_amp = target_amp.to(device)

        # cropping to desired region for loss
        if homography_res is not None:
            target_res = homography_res
        else:
            target_res = target_res[0]  # use resolution of first image in batch

        optimizer.zero_grad()

        # forward model
        slm_amp, slm_phase = phase_generator(target_amp)
        output_complex = model_prop(slm_phase)

        if loss_fun == 2:
            output_complex = output_complex * scaleLoss

        output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)

        output_amp = torch.pow(output_lin_intensity, 0.5)
         # VGG assumes RGB input, we just replicate
        if loss_fun == 3:
            target_amp = target_amp.repeat(1, 3, 1, 1)
            output_amp = output_amp.repeat(1, 3, 1, 1)

        # ignore the cropping and do full-image loss
        if 'nocrop' in run_id:
            loss_nocrop = loss(output_amp, target_amp)

        # crop outputs to the region we care about
        output_amp = utils.crop_image(output_amp, target_res, stacked_complex=False)
        target_amp = utils.crop_image(target_amp, target_res, stacked_complex=False)

        # force equal mean amplitude when checking loss
        if 'force_scale' in run_id:
            print('scale forced equal', end=' ')
            with torch.no_grad():
                scaled_out = output_amp * target_amp.mean() / output_amp.mean()
            output_amp = output_amp + (scaled_out - output_amp).detach()

        # loss and optimize
        loss_main = loss(output_amp, target_amp)
        if warm_start or opt.disable_loss_amp:
            loss_amp = 0
        else:
            # extra loss term to encourage uniform amplitude at SLM plane
            # helps some networks converge properly initially
            loss_amp = mse_loss(slm_amp.mean(), slm_amp)

        loss_val = ((loss_nocrop if 'nocrop' in run_id else loss_main)
                    + 0.1 * loss_amp)

        loss_tv = tv_loss(slm_phase)
        loss_all = loss_val + loss_tv

        loss_all.backward()
        optimizer.step()
        if opt.step_lr:
            scheduler.step()

        # print and output to tensorboard
        ik = k + i * len(image_loader)
        if use_mse_init and i >= num_mse_epochs:
            ik += num_mse_iters - len(image_loader)
        print(f'iteration {ik}: {loss_all.item()}')

        with torch.no_grad():
            writer.add_scalar('Loss', loss_all, ik)

            if ik % 50 == 0:
                # write images and loss to tensorboard
                writer.add_image('Target Amplitude', target_amp[0, ...], ik)

                # normalize reconstructed amplitude
                output_amp0 = output_amp[0, ...]
                maxVal = torch.max(output_amp0)
                minVal = torch.min(output_amp0)
                tmp = (output_amp0 - minVal) / (maxVal - minVal)
                writer.add_image('Reconstruction Amplitude', tmp, ik)

                # normalize SLM phase
                writer.add_image('SLM Phase', (slm_phase[0, ...] + math.pi) / (2 * math.pi), ik)

            if loss_fun == 2:
                writer.add_scalar('Scale factor', scaleLoss, ik)
    # scheduler.step()
    # save trained model
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(phase_generator.state_dict(),
               f'checkpoints/unet_fft/{run_id}_{chan_str}_{i+1}.pth')
