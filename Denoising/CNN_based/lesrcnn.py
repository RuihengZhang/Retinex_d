import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from sewar.full_ref import vifp
from scipy.sparse.linalg import spsolve
import random
import sys
import argparse
import math
import torch.nn.functional as F
# 定义模型
def init_weights(modules):
    pass
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0) #3 is size of output, 3 is size of input, 1 is kernel 1 is padding, 0 is group 
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1) # view(3,3,1,1) convert a shape into (3,3,1,1) eye(3) is a 3x3 matrix and diagonal is 1.
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        #in_channels, out_channels,ksize=3, stride=1, pad=1
        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out
class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out
class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)
class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                #modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            #modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(64, 64, group=group)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3      
class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv16 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv17_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv18 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=padding,groups=groups,bias=False))
        '''
        self.conv18 =  nn.Conv2d(in_channels=features,out_channels=features1,kernel_size=kernel_size1,padding=padding1,groups=groups,bias=False)
        self.ReLU = nn.ReLU(inplace=True)
        '''
        self.ReLU=nn.ReLU(inplace=True)
        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)

    def forward(self, x, scale=2):
        #print '-------dfd'
        x = self.sub_mean(x)
        c0 = x
        x1 = self.conv1(x)
        x1_1 = self.ReLU(x1)
        x2 = self.conv2(x1_1)
        x3 = self.conv3(x2)
        x2_3 = x1+x3
        x2_4 = self.ReLU(x2_3)
        x4 = self.conv4(x2_4)
        x5 = self.conv5(x4)
        x3_5 = x2_3 + x5 
        x3_6 = self.ReLU(x3_5)
        x6 = self.conv6(x3_6)
        x7 = self.conv7(x6)
        x7_1 = x3_5 + x7 
        x7_2 = self.ReLU(x7_1)
        x8 = self.conv8(x7_2)
        x9 = self.conv9(x8)
        x9_2 = x7_1 + x9
        x9_1 = self.ReLU(x9_2)
        x10 = self.conv10(x9_1)
        x11 = self.conv11(x10)
        x11_1 = x9_2 + x11
        x11_2 = self.ReLU(x11_1)
        x12 = self.conv12(x11_2)
        x13 = self.conv13(x12)
        x13_1 = x11_1 + x13
        x13_2 = self.ReLU(x13_1)
        x14 = self.conv14(x13_2)
        x15 = self.conv15(x14)
        x15_1 = x15+x13_1
        x15_2 = self.ReLU(x15_1)
        x16 = self.conv16(x15_2)
        x17 = self.conv17(x16)
        x17_2 = x17 + x15_1 
        x17_3 = self.ReLU(x17_2)
        temp = self.upsample(x17_3, scale=scale)
        x1111 = self.upsample(x1_1, scale=scale) #tcw
        temp1 = x1111+temp #tcw
        temp2 = self.ReLU(temp1)
        temp3 = self.conv17_1(temp2)
        temp4 = self.conv17_2(temp3)
        temp5 = self.conv17_3(temp4)
        temp6 = self.conv17_4(temp5)
        x18 = self.conv18(temp6)
        out = self.add_mean(x18)
        #out = x18
        return out
    
parser = argparse.ArgumentParser()
parser.add_argument('--need_resize', type=bool, default=False)  # 是否需要resize, 如果需要resize，一律改到256大小
parser.add_argument('--img_names', type=list[str], default=random.sample(os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'), 2785)) # 获取打乱顺序的文件列表
parser.add_argument('--I_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I')
parser.add_argument('--S_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S')
parser.add_argument('--E_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E')
parser.add_argument('--py_filename', type=str, default=os.path.splitext(os.path.basename(os.path.abspath(sys.argv[0])))[0])
parser.add_argument('--img_type', type=str, default="RGB")

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
    model = Net()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 80 if decompose_mode == 'retinexd' else 25
    dataset = VariableDataset(decompose_mode=decompose_mode, denoising_mode=denoising_mode)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
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
            if epoch+1 == epochs or epoch == 0:
                torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/{args.py_filename}/{args.py_filename}_{denoising_mode}.pth')
        else:
            for degraded_I, E, S, groundtruth_I in progress_bar:
                degraded_I, E, S, groundtruth_I = degraded_I.to(device), E.to(device), S.to(device), groundtruth_I.to(device)
                optimizer.zero_grad()
                enhanced_I, enhanced_S = model(degraded_I), model(S)
                loss = criterion(E*enhanced_S, groundtruth_I) + criterion(enhanced_I, E*enhanced_S)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or epoch == 0:
                save_name = f'{args.py_filename}_{denoising_mode}_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/{args.py_filename}/', save_name))    

# 定义测试函数
def lesrcnn(img, decompose_mode=None):
    model = Net()
    if decompose_mode == None:
        model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/lesrcnn/lesrcnn_0to1.pth'))
    else:
        ckt_path = r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/lesrcnn/lesrcnn_0to1_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) # 加载图片
    with torch.no_grad():
        denoised_img = model(img)
    rgb_image = np.transpose(np.array(denoised_img.squeeze()), (1, 2, 0))
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
   



if __name__ == '__main__':
    # variabletrain(device='cuda:1', decompose_mode='retinexd', denoising_mode='2to3')
    model = Net()
    print(model(torch.randn(1, 3, 360, 500)).shape)




