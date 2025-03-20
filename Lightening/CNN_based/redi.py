import torch
from torch.nn.utils import weight_norm as wn
import torch.nn as nn
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import argparse
import random
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn

################### ICNR initialization for pixelshuffle        
def ICNR(tensor, upscale_factor=2, negative_slope=1, fan_type='fan_in'):
    
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    nn.init.kaiming_normal_(subkernel, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel

def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=4, num_classes=3, weight_normalization = True):

    layers = []
    
    if bn:
        m = nn.BatchNorm2d(inc)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)
    
    if activation=='before':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        
    if pixelshuffle_init:
        m = nn.Conv2d(in_channels=inc, out_channels=num_classes * (upscale ** 2),
                                  kernel_size=3, padding = 3//2, groups=1, bias=True, stride=1)
        nn.init.constant_(m.bias, 0)
        with torch.no_grad():
            kernel = ICNR(m.weight, upscale, negative_slope, fan_type)
            m.weight.copy_(kernel)
    else:
        m = nn.Conv2d(in_channels=inc, out_channels=outc,
     kernel_size=kernel_size, padding = (kernel_size-1)//2, groups=groups, bias=bias, stride=1)
        init_gain = 0.02
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight, gain = init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)
    
    if activation=='after':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
            
    return nn.Sequential(*layers)
    
    
class ResBlock(nn.Module):    
    def __init__(self,in_c):
        super(ResBlock, self).__init__()
                
        self.conv1 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv2 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x
        
        
class make_dense(nn.Module):
    
    def __init__(self, nChannels=64, growthRate=32, pos=False):
        super(make_dense, self).__init__()
        
        kernel_size=3
        if pos=='first':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        elif pos=='middle':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        elif pos=='last':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        else:
            raise NotImplementedError('ReLU position error in make_dense')
        
    def forward(self, x):
        return torch.cat((x, self.conv(x)), 1)
        
        
        
class RDB(nn.Module):
    def __init__(self, nChannels=96, nDenselayer=5, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        
        modules.append(make_dense(nChannels_, growthRate, 'first'))
        nChannels_ += growthRate
        for i in range(nDenselayer-2):    
            modules.append(make_dense(nChannels_, growthRate, 'middle'))
            nChannels_ += growthRate 
        modules.append(make_dense(nChannels_, growthRate, 'last'))
        nChannels_ += growthRate
        
        self.dense_layers = nn.Sequential(*modules)
            
        self.conv_1x1 = conv_layer(nChannels_, nChannels, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
    
    def forward(self, x):
        return self.conv_1x1(self.dense_layers(x)) + x
        
        
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        
        self.up4 = nn.PixelShuffle(4)
        self.up2 = nn.PixelShuffle(2)
        
        self.conv32x = nn.Sequential(        
                        conv_layer(1024, 128, kernel_size=3, groups=128, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        conv_layer(128, 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
                        )
                        
        self.RDB1 = RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=5, growthRate=32)
        self.RDB3 = RDB(nChannels=64, nDenselayer=5, growthRate=32)

        self.rdball = conv_layer(int(64*3), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv_rdb8x = conv_layer(int(64//16), 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.resblock8x = ResBlock(64)
        
        self.conv32_8_cat = nn.Sequential(
                        conv_layer(128, 32, kernel_size=3, groups=4, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        conv_layer(32, 192, kernel_size=3, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        self.up4)                      
        
        
        self.conv2x = conv_layer(4, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv_2_8_32 = conv_layer(24, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        
    def downshuffle(self,var,r):
        b,c,h,w = var.size()
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w).contiguous()
        
        
    def forward(self,low):
            
        low2x = self.downshuffle(low,2)
        
        # 32x branch starts
        low32x_beforeRDB = self.conv32x(self.downshuffle(low2x,16))
        rdb1 = self.RDB1(low32x_beforeRDB)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb8x = torch.cat((rdb1,rdb2,rdb3),dim=1)
        rdb8x = self.rdball(rdb8x)+low32x_beforeRDB
        rdb8x = self.up4(rdb8x)
        rdb8x = self.conv_rdb8x(rdb8x)
        
        # 8x branch starts
        low8x = self.resblock8x(self.downshuffle(low2x,4))
        cat_32_8 = torch.cat((low8x,rdb8x),dim=1).contiguous()
        
        b,c,h,w = cat_32_8.size()
        G=2
        cat_32_8 = cat_32_8.view(b, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        cat_32_8 = self.conv32_8_cat(cat_32_8)
        
        
        # 2x branch starts
        low2x = torch.cat((self.conv2x(low2x),cat_32_8),dim=1)
        low2x = self.up2(self.conv_2_8_32(low2x))
        out = torch.clamp(low2x,min=0.0, max=1.0)
        return torch.mean(out, dim=1, keepdim=True)
    
parser = argparse.ArgumentParser()
parser.add_argument('--need_resize', type=bool, default=True)  # 是否需要resize, 如果需要resize，一律改到256大小
parser.add_argument('--img_names', type=list[str], default=random.sample(os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'), 2785)) # 获取打乱顺序的文件列表
parser.add_argument('--I_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I')
parser.add_argument('--S_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S')
parser.add_argument('--E_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E')
parser.add_argument('--py_filename', type=str, default=os.path.splitext(os.path.basename(os.path.abspath(sys.argv[0])))[0])
parser.add_argument('--img_type', type=str, default="L")
args = parser.parse_args()


class VariableDataset(Dataset): 
    def __init__(self, decompose_mode, denoising_mode):
        self.img_names = args.img_names[0:2735]  # 取前2735张图片作为训练集
        self.decompose_mode = decompose_mode
        self.denoising_mode = denoising_mode
        if args.need_resize:
            self.transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()
    def __len__(self):
        return 2735
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        degraded_I_path = os.path.join(args.I_folder_path, 'I_'+self.denoising_mode[-1], img_name)
        groundtruth_I_path = os.path.join(args.I_folder_path, 'I_'+self.denoising_mode[0], img_name)
        if self.decompose_mode == None:
            return self.transform(Image.open(degraded_I_path).convert(args.img_type)), self.transform(Image.open(groundtruth_I_path).convert(args.img_type))   # I1, I0
        if self.decompose_mode == 'retinexd':
            S_img_path = os.path.join(args.S_folder_path, 'S_'+self.denoising_mode[-1], img_name)
            E_img_path = os.path.join(args.E_folder_path, 'E_'+self.denoising_mode[-1], img_name)
            return self.transform(Image.open(degraded_I_path).convert(args.img_type)), self.transform(Image.open(E_img_path).convert(args.img_type)), self.transform(Image.open(S_img_path).convert(args.img_type)), self.transform(Image.open(groundtruth_I_path).convert(args.img_type))   # I1, E1, S1, I0

def variabletrain(device, decompose_mode, denoising_mode): 
    model = Net()     # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60 if decompose_mode == 'retinexd' else 30
    dataset = VariableDataset(decompose_mode=decompose_mode, denoising_mode=denoising_mode)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        if decompose_mode is None:
            for degraded_I, groundtruth_I in progress_bar:
                degraded_I, groundtruth_I = degraded_I.to(device), groundtruth_I.to(device)
                optimizer.zero_grad()
                enhanced_I = model(degraded_I)
                loss = criterion(enhanced_I, groundtruth_I)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 5) == 0:
                torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/{args.py_filename}/{args.py_filename}_{denoising_mode}.pth')
        else:
            for degraded_I, E, S, groundtruth_I in progress_bar:
                degraded_I, E, S, groundtruth_I = degraded_I.to(device), E.to(device), S.to(device), groundtruth_I.to(device)
                optimizer.zero_grad()
                enhanced_I, enhanced_E = model(degraded_I), model(E)
                loss = criterion(S*enhanced_E, groundtruth_I) + criterion(enhanced_I, S*enhanced_E)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 5) == 0:
                save_name = f'{args.py_filename}_{denoising_mode}_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/{args.py_filename}/', save_name))

def redi(img, decompose_mode=None):
    model = Net() 
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/redi/redi_0to2.pth'))
    else:
        ckt_path = f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/redi/redi_0to2_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    input_array = img
    tensor = torch.tensor(input_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) # tensor: [1, 1, 360, 500]
    # repeated_tensor = tensor.repeat(1, 3, 1, 1)
    resized_tensor = torch.nn.functional.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    with torch.no_grad():
        tensor = model(resized_tensor)
    deresized_tensor = torch.nn.functional.interpolate(tensor, size=(360, 500), mode='bilinear', align_corners=False)
    derepeated_tensor = deresized_tensor.squeeze()
    
    return derepeated_tensor.numpy()


    

if __name__ == '__main__':
    # model = Net()
    # x = torch.randn(1, 1, 256, 256)
    # y = model(x)
    # print(y.shape)
    variabletrain('cuda:1', decompose_mode='retinexd', denoising_mode='0to2')