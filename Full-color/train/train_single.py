import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Gray Hologram Generation')).parse_args()
print(opt)
import torch
device = torch.device('cuda')
import utils
from dataset.dataset_single import *
from losses import VGGLoss
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
import glob
import random
import time
import numpy as np
import datetime
from pdb import set_trace as stx
from utils.dir_utils import mkdir
from utils.model_utils import get_arch
from pytorch_msssim import ms_ssim
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from reconstruction import holo_propagator
from loss.loss_provider import LossProvider
######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'single_channel', opt.color, opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
mkdir(result_dir)
mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
phase_generator = get_arch(opt)

# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# Propagation parameters
prop_dist = (7 * cm, 7 * cm, 7 * cm)[opt.channel]
wavelength = (680 * nm, 520 * nm, 450 * nm)[opt.channel]
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
homography_res = (880, 1600)
model_prop = holo_propagator(wavelength, prop_dist, feature_size)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(phase_generator)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(phase_generator.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(phase_generator.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
phase_generator = torch.nn.DataParallel(phase_generator) 
phase_generator.to(device)
     

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from " + path_chk_rest)
    utils.load_checkpoint(phase_generator, path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

######### Loss ###########
## MSE Loss
criterion_MSE = nn.MSELoss(size_average=True).to(device)
## MS-SSIM Loss
criterion_SSIM = ms_ssim
## VGG Loss
criterion_VGG = VGGLoss(device)
## Watson-fft Loss
provider = LossProvider()
criterion_WFFT = provider.get_loss_function('watson-fft', deterministic=False, colorspace='grey',
                                            pretrained=True, reduction='sum').to(device)

######### DataLoader ###########
print('===> Loading datasets')

train_loader = get_training_data(opt)
val_loader = get_validation_data(opt)

len_trainset = train_loader.__len__()
len_valset = val_loader.__len__()
print("Size of training set: ", len_trainset,", size of validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
loss_scaler = NativeScaler()
# torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_ssim_loss = 0
    epoch_vgg_loss = 0
    epoch_wfft_loss = 0
    train_id = 1

    for i, target in enumerate(train_loader, 0): 
        # zero_grad
        optimizer.zero_grad()

        # get target image
        target_amp, target_res, target_filename = target

        target_amp = target_amp.to(device)
        
        # Hologram generation
        slm_amp, slm_phase = phase_generator(target_amp)
        
        # ASM propagation
        recon_amp = model_prop(slm_phase)
        recon_amp = torch.sqrt(torch.pow(recon_amp, 2) * 0.95)

        # Crop ROI
        recon_amp = utils.crop_image(recon_amp, homography_res, stacked_complex=False)
        target_amp = utils.crop_image(target_amp, homography_res, stacked_complex=False)

        # Loss function
        mse_loss = opt.w_mse * criterion_MSE(recon_amp, target_amp)
        ssim_loss = opt.w_ssim * (1 - criterion_SSIM(recon_amp, target_amp, data_range=1))
        vgg_loss = opt.w_vgg * criterion_VGG(recon_amp, target_amp)
        wfft_loss = opt.w_wfft * criterion_WFFT(recon_amp, target_amp)     

        loss = mse_loss + ssim_loss + vgg_loss + wfft_loss

        loss_scaler(
                loss, optimizer,parameters=phase_generator.parameters())
        
        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_ssim_loss += ssim_loss.item()
        epoch_vgg_loss  += vgg_loss.item()
        epoch_wfft_loss += wfft_loss.item()

        #### Evaluation ####
        if ( epoch + 1 ) % opt.val_frequency == 0 and epoch > 0:
            with torch.no_grad():
                phase_generator.eval()
                val_mse = []
                val_ssim = []
                val_wfft = []
                for ii, data_val in enumerate((val_loader), 0):

                    target_amp, target_res, target_filename = data_val

                    target_amp = target_amp.to(device)

                    # Hologram generation
                    slm_amp, slm_phase = phase_generator(target_amp)
        
                    # ASM propagation
                    recon_amp = model_prop(slm_phase)
                    recon_amp = torch.sqrt(torch.pow(recon_amp, 2) * 0.95)

                    # Crop ROI
                    recon_amp = utils.crop_image(recon_amp, homography_res, stacked_complex=False)
                    target_amp = utils.crop_image(target_amp, homography_res, stacked_complex=False)

                    # for speeding up test and saving memory cost, turn off vgg loss in test.
                    vgg_loss = torch.tensor([0.0]).to(device) 
                    mse_loss = opt.w_mse * criterion_MSE(recon_amp, target_amp)
                    ssim_loss = opt.w_ssim * (1 - criterion_SSIM(recon_amp, target_amp, data_range=1))
                    wfft_loss = opt.w_wfft * criterion_WFFT(recon_amp, target_amp) 

                    val_mse.append(mse_loss.item())
                    val_ssim.append(ssim_loss.item())
                    val_wfft.append(wfft_loss.item())

                val_mse = sum(val_mse)/len_valset   
                val_ssim = sum(val_ssim)/len_valset
                val_wfft = sum(val_wfft)/len_valset

                print("[Ep %d val_mse %.4f\t val_ssim %.4f\t val_wfft: %.4f\t] " % (epoch, val_mse, val_ssim, val_wfft))
                with open(logname,'a') as f:
                    f.write("[Ep %d val_mse %.4f\t val_ssim %.4f\t val_wfft: %.4f\t] " \
                        % (epoch, val_mse, val_ssim, val_wfft)+'\n')
                phase_generator.train()
                # torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tMSE: {:.4f}\tSSIM: {:.4f}\tVGG: {:.4f}\tWFFT: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, epoch_mse_loss, epoch_ssim_loss,epoch_vgg_loss,epoch_wfft_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tMSE: {:.4f}\tSSIM: {:.4f}\tVGG: {:.4f}\tWFFT: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, epoch_mse_loss, epoch_ssim_loss,epoch_vgg_loss,epoch_wfft_loss,scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': phase_generator.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': phase_generator.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ", datetime.datetime.now().isoformat())
