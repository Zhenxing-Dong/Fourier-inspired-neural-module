import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr


from model import *
from model_fft import *

def get_arch(opt):

    arch = opt.arch

    # units
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


    print('You choose' + arch + '...')
    if arch == 'UNet_single':
        phase_restoration = InitialPhaseUnet(4, 16)

    elif arch == 'HoloNet_single':

        # Propagation parameters
        prop_dist = (7 * cm, 7 * cm, 7 * cm)[opt.channel]
        wavelength = (680 * nm, 520 * nm, 450 * nm)[opt.channel]
        feature_size = (3.74 * um, 3.74 * um)  # SLM pitch

        phase_restoration = HoloNet_single(distance=prop_dist,
                                           wavelength=wavelength,
                                           feature_size=feature_size,
                                           initial_phase=InitialPhaseUnet(4, 16, num_in=1, num_out=1),
                                           final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2, num_out=1))
        
    elif arch == 'HoloNet':

        prop_dist = 7 * cm
        wavelengths = (680 * nm, 520 * nm, 450 * nm)
        feature_size = (3.74 * um, 3.74 * um)

        phase_restoration = HoloNet_full(distance=prop_dist,
                                         wavelengths=wavelengths,
                                         feature_size=feature_size,
                                         initial_phase=InitialPhaseUnet(4, 16, num_in=3, num_out=3),
                                         final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=6, num_out=3))   

    elif arch == 'UNet':
        phase_restoration = UNet()  

    elif arch == 'UNet_fft':
        phase_restoration = UNet_fft()      

    elif arch == 'HoloNet_fft':

        prop_dist = 7 * cm
        wavelengths = (680 * nm, 520 * nm, 450 * nm)
        feature_size = (3.74 * um, 3.74 * um)

        phase_restoration = HoloNet_wfft_full(distance=prop_dist,
                                              wavelengths=wavelengths,
                                              feature_size=feature_size,
                                              initial_phase=UNet_fft(dim=16),
                                              final_phase_only=PhaseOnlyUnet(4, 16, num_in=6, num_out=3))             


        
    else:
        raise Exception("Arch error!")

    return phase_restoration