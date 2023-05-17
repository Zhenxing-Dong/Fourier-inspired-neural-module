import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--nepoch', type=int, default=50, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='DIV2K')
        parser.add_argument('--pretrain_weights',type=str, default='./log/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default ='HoloNet_single',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='single',  help='hologram generation mode')
        parser.add_argument('--val_frequency', type=int, default=100, help='run validation every x iterations')

        # args for propagation
        parser.add_argument('--channel', type=int, default=1, help='Red:0, Green:1, Blue:2, Full color:None')
        parser.add_argument('--color', type=str, default='green', help='Red:0, Green:1, Blue:2, Full color:3')

        # args for loss function
        parser.add_argument('--w_mse', type=float, default=1.0, help='the weight parameter for MSE loss term')
        parser.add_argument('--w_vgg', type=float, default=0.025, help='the balancing weight for VGG perceptual loss')
        parser.add_argument('--w_ssim', type=float, default=0.05, help=' the balancing weight for MS-SSIM loss')
        parser.add_argument('--w_wfft', type=float, default=1e-8, help='the balancing weihgt for Watson-FFT loss')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=1, help='checkpoint')
       
        # args for training
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--data_dir', type=str, default ='./datasets/DIV2K/',  help='dir of data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        return parser
