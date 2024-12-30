# Fourier-inspired-neural-module
Fourier-inspired neural module for real-time and high-fidelity computer-generated holography

## Get started
Please integrate the [train_fft_tv.py](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/blob/master/train_fft_tv.py), [main.py](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/blob/master/main.py), [eval.py](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/blob/master/eval.py), [unet_fft.py](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/blob/master/unet_fft.py) and [holonet_fft.py](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/blob/master/holonet_fft.py) into [neural holography](https://github.com/computational-imaging/neural-holography).

## Train
#### UNet with proposed module
    python train_fft_tv.py --run_id=unet_fft --perfect_prop_model=True --purely_unet=True --batch_size=1 --channel=1 
#### HoloNet with proposed module
    python train_fft_tv.py --run_id=holonet_fft --perfect_prop_model=True --purely_unet=False --batch_size=1 --channel=1 
    
## Test
#### Pretrained model
Download the pretrained model weights from [here](https://github.com/Zhenxing-Dong/Fourier-inspired-neural-module/tree/master/pretrained_network)

#### UNet with proposed module
    python main.py --channel=1 --method=UNET_fft --root_path=./phases --generator_dir=./pretrained_networks
#### HoloNet with proposed module   
    python main.py --channel=1 --method=HOLONET_inital_fft --root_path=./phases --generator_dir=./pretrained_networks

## Eval

#### UNet with proposed module
    python eval.py --channel=1 --root_path=./phases/_UNET_fft_ASM --prop_model=ASM
#### HoloNet with proposed module   
    python eval.py --channel=1 --root_path=./phases/_HOLONET_inital_fft_ASM --prop_model=ASM
 
## Acknowledgement
The codes are built on [neural holography](https://github.com/computational-imaging/neural-holography). We sincerely appreciate the authors for sharing their codes.
