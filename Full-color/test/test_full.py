import os,sys
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import numpy as np
import argparse
from tqdm import tqdm
import scipy.io as sio
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset.dataset_full import *
from utils.model_utils import get_arch, load_checkpoint
from reconstruction import full_holo_propagator
import imageio
import cv2
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))



parser = argparse.ArgumentParser(description='Single channel Hologram generation')
parser.add_argument('--data_dir', default='./datasets/DIV2K/', type=str, help='Directory of validation images')
parser.add_argument('--recon_dir', default='./results/recon/',      type=str, help='Directory for results')
parser.add_argument('--pred_phase_dir', default='./results/phase/',      type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/full_channel/rgb/HoloNet_fft/model_epoch_42.pth',   type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='UNet_fft', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--channel', type=int, default=2, help='Red:0, Green:1, Blue:2, Full color:3')
parser.add_argument('--color', type=str, default='rgb', help='Red:0, Green:1, Blue:2, Full color:3')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

device = torch.device('cuda')
phase_generator= get_arch(args)
load_checkpoint(phase_generator, args.weights)
print("===>Testing using weights: ", args.weights)
phase_generator.to(device)
phase_generator.eval()

utils.cond_mkdir(os.path.join(args.recon_dir, args.color))
utils.cond_mkdir(os.path.join(args.pred_phase_dir, args.color))
# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# Propagation parameters
prop_dist = 7 * cm
wavelengths = (680 * nm, 520 * nm, 450 * nm)
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
homography_res = (880, 1600)
model_prop = full_holo_propagator(wavelengths, prop_dist, feature_size)
test_loader = get_test_data(args)

psnrs = {'amp': [], 'lin': [], 'srgb': []}
ssims = {'amp': [], 'lin': [], 'srgb': []}
idxs = []
with torch.no_grad():
    for i, data_test in enumerate(test_loader, 0):
        # get target image
        target_amp, target_res, target_filename = data_test
        target_amp = target_amp.to(device)
        target_path, target_filename = os.path.split(target_filename[0])
        target_idx = target_filename.split('_')[-1]
        print(f'    - running for img_{target_idx}...')

        # Hologram generation
        pred_phase = phase_generator(target_amp)

        recon_amp = model_prop(pred_phase)

        # crop to ROI
        target_amp = utils.crop_image(target_amp, target_shape=homography_res, stacked_complex=False).to(device)

        recon_amp = utils.crop_image(recon_amp, target_shape=homography_res, stacked_complex=False)

        # normalization
        recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                       / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))
        
        # tensor to numpy
        recon_amp = recon_amp.squeeze().cpu().detach().numpy()
        target_amp = target_amp.squeeze().cpu().detach().numpy()
        pred_phase = pred_phase.squeeze().cpu().detach().numpy()

        recon_amp = recon_amp.transpose(1, 2, 0)
        target_amp = target_amp.transpose(1, 2, 0)
        pred_phase = pred_phase.transpose(1, 2, 0)

        # calculate metrics
        psnr_val, ssim_val = utils.get_psnr_ssim(recon_amp, target_amp, multichannel=True)

        idxs.append(target_idx)

        for domain in ['amp', 'lin', 'srgb']:
            psnrs[domain].append(psnr_val[domain])
            ssims[domain].append(ssim_val[domain])
            print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ')


        # save reconstructed image in srgb domain
        recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp**2, 0.0, 1.0))
        target_srgb = utils.srgb_lin2gamma(np.clip(target_amp**2, 0.0, 1.0))

        phase_out_8bit = utils.phasemap_8bit(pred_phase, inverted=True)
        
        imageio.imwrite(os.path.join(args.recon_dir, args.color, args.arch, f'{target_idx}_{args.color}.png'), (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))
        
        imageio.imwrite(os.path.join(args.pred_phase_dir, args.color, args.arch, 'red', f'{target_idx}_red.png'), phase_out_8bit[:, :, 0])
        imageio.imwrite(os.path.join(args.pred_phase_dir, args.color, args.arch, 'green', f'{target_idx}_green.png'), phase_out_8bit[:, :, 1])
        imageio.imwrite(os.path.join(args.pred_phase_dir, args.color, args.arch, 'blue', f'{target_idx}_blue.png'), phase_out_8bit[:, :, 2])

# save it as a .mat file
data_dict = {}
data_dict['img_idx'] = idxs
for domain in ['amp', 'lin', 'srgb']:
    data_dict[f'ssims_{domain}'] = ssims[domain]
    data_dict[f'psnrs_{domain}'] = psnrs[domain]

sio.savemat(os.path.join(args.recon_dir, args.color, args.arch, f'metrics_{args.color}.mat'), data_dict)

