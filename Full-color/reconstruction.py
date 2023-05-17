import numpy as np
import torch
import torch.nn.functional as F
import propagation_utils as utils 
from propagation_ASM import propagation_ASM, full_propagation_ASM

def np_circ_filter(batch,
                   num_channels,
                   res_h,
                   res_w,
                   filter_radius,
                   ):
    """create a circular low pass filter
    """
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = x**2+y**2 <= filter_radius**2
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    return np_filter


class holo_propagator(torch.nn.Module):
    '''the class used for calculate the wave propagation'''
    def __init__(self, wavelength, prop_dist, feature_size=(3.74e-6, 3.74e-6), precomped_H=None):
        super(holo_propagator, self).__init__()
        self.precomped_H = precomped_H  # precomputed H matrix in the ASM propagation formula
        self.wavelength = wavelength  # the wavelength that will be used during the diffraction calculation
        self.prop_dist = prop_dist # propagation distance (here we give it in meters)
        self.feature_size = feature_size # the pixel pitch size (in meters)
        self.propagator = propagation_ASM  # the function used for calculating the wavefield after propagation
    
    def forward(self, input_phase):
        # slm_phase = input_phase 
        # real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase) # transform amp-phase representation to real-image representation
        # slm_field = torch.stack((real, imag), -1)  # since the codes are based on pytorch 1.6, the complex tensor is represented as [B, C, H, W, 2]
        slm_field = input_phase

        recon_field = utils.propagate_field(slm_field, self.propagator, self.prop_dist, self.wavelength, self.feature_size,
                                            prop_model='ASM', dtype = torch.float32)

        # get the amplitude map from the propagated wavefield
        # recon_amp_c, _ = utils.rect_to_polar(recon_field[..., 0], recon_field[..., 1])  
        recon_amp_c = recon_field.abs()
        return recon_amp_c     


class full_holo_propagator(torch.nn.Module):
    '''the class used for calculate the wave propagation'''
    def __init__(self, wavelengths, prop_dist, feature_size=(3.74e-6, 3.74e-6), precomped_H=None):
        super(full_holo_propagator, self).__init__()
        self.precomped_H = precomped_H  # precomputed H matrix in the ASM propagation formula
        self.wavelengths = wavelengths  # the wavelength that will be used during the diffraction calculation
        self.prop_dist = prop_dist # propagation distance (here we give it in meters)
        self.feature_size = feature_size # the pixel pitch size (in meters)
        self.propagator = propagation_ASM  # the function used for calculating the wavefield after propagation
        self.color = [0,1,2]
    
    def forward(self, input_phase):
        slm_phase = input_phase 

        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase) 
        slm_field = torch.complex(real, imag)

        recon = []
        for wv, i in zip(self.wavelengths, self.color):

            target_field = slm_field[:, i, :, :].unsqueeze(1)
            recon_field = utils.propagate_field(target_field, self.propagator, self.prop_dist, wv, self.feature_size,
                                                prop_model='ASM', dtype = torch.float32)
            
            recon.append(recon_field)
        recon = torch.cat(recon, dim=1)
        # get the amplitude map from the propagated wavefield
        recon_amp_c = recon.abs()
        return recon_amp_c 

if __name__ == "__main__":
    """a small piece of codes for testing the holo propagator class"""

    import cv2
    import skimage.io
    import os
    import imageio
    import torch.fft as tfft

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    wavelengths = (638 * 1e-9, 520 * 1e-9, 450 * 1e-9)
    propagator = full_holo_propagator(wavelengths, prop_dist=1 * 1e-3, feature_size=(8e-6, 8e-6))
    propagator_red = holo_propagator(638 * 1e-9, prop_dist=1 * 1e-3, feature_size=(8e-6, 8e-6))
    propagator_green = holo_propagator(520 * 1e-9, prop_dist=1 * 1e-3, feature_size=(8e-6, 8e-6))
    propagator_blue = holo_propagator(450 * 1e-9, prop_dist=1 * 1e-3, feature_size=(8e-6, 8e-6))


    dtype = torch.float32  
    device = torch.device('cuda') 

    circ_filter = np_circ_filter(1, 1, 1080, 1920, 1080//2)
    circ_filter = torch.tensor(circ_filter).to(device)


    red = skimage.io.imread('./red.png') / 255.
    red = torch.from_numpy(red * 2 * np.pi - np.pi).float().to(device)
    red = red[None,None,:,:]

    green = skimage.io.imread('./green.png') / 255.
    green = torch.from_numpy(green * 2 * np.pi - np.pi).float().to(device)
    green = green[None,None,:,:]

    blue = skimage.io.imread('./blue.png') / 255.
    blue = torch.from_numpy(blue * 2 * np.pi - np.pi).float().to(device)
    blue = blue[None,None,:,:]


    # phase_only = torch.cat([red, green, blue], dim=1)

    real_red, imag_red = utils.polar_to_rect(torch.ones_like(red), red)
    slm_field_red = torch.complex(real_red, imag_red)
    slm_field_red = tfft.fftshift(tfft.fftn(slm_field_red, dim=(-2, -1), norm='ortho'), (-2, -1))
    slm_field_red = slm_field_red * circ_filter
    slm_field_red = tfft.ifftn(tfft.ifftshift(slm_field_red, (-2, -1)), dim=(-2, -1), norm='ortho')


    # real, imag = utils.polar_to_rect(amp, phase)
    # slm_field = torch.complex(real, imag)
    
    recon_amp_red = propagator_red(slm_field_red)

    recon_amp_red = recon_amp_red.squeeze().cpu().detach().numpy()

    # recon_amp_color = recon_amp_color.transpose(1, 2, 0)

    recon_srgb_red = utils.srgb_lin2gamma(np.clip(recon_amp_red**2, 0.0, 1.0))

    
    imageio.imwrite('./precon_red.png', (recon_srgb_red * np.iinfo(np.uint8).max).round().astype(np.uint8))

    