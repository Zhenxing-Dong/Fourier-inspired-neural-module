# Fourier-inspired-neural-module
Fourier-inspired neural module for real-time and high-fidelity computer-generated holography

## Get started
Please integrate the train_fft_tv.py, main.py, unet_fft.py and holonet_fft.py into [neural holography](https://github.com/computational-imaging/neural-holography)

## Train

#### UNet with proposed module
    python train_fft_tv.py --run_id=unet_fft --perfect_prop_model=True --purely_unet=True --batch_size=1 --channel=1 
#### HoloNet with proposed module
    python train_fft_tv.py --run_id=holonet_fft --perfect_prop_model=False --purely_unet=True --batch_size=1 --channel=1 
    
## Test

#### Testing dataset
Download the testing set in our paper is [here](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/tree/master/test_data)
#### Pretrained model
Download the pretrained model weights from [here](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/tree/master/pretrained_network)

#### UNet with proposed module
    python main.py --channel=1 --method=UNET_fft --root_path=./phases --generator_dir=./pretrained_networks
#### HoloNet with proposed module   
    python main.py --channel=1 --method=HOLONET_inital_fft --root_path=./phases --generator_dir=./pretrained_networks
    
  
