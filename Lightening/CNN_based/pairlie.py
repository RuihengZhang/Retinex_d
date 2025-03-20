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
class N_net(nn.Module):
    def __init__(self, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.N_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        

        self.N_net = N_net(num=64)        

    def forward(self, input):
        x = self.N_net(input)

        return x
    
parser = argparse.ArgumentParser()
parser.add_argument('--need_resize', type=bool, default=False)  # 是否需要resize, 如果需要resize，一律改到256大小
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
    model = net()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60 if decompose_mode == 'retinexd' else 30
    dataset = VariableDataset(decompose_mode=decompose_mode, denoising_mode=denoising_mode)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
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

def pairlie(img, decompose_mode=None):
    model = net()
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/pairlie/pairlie_0to2.pth'))
    else:
        ckt_path = f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/pairlie/pairlie_0to2_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    input_array = img
    tensor = torch.tensor(input_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) # tensor: [1, 1, 360, 500]
    # repeated_tensor = tensor.repeat(1, 3, 1, 1)
    # resized_tensor = torch.nn.functional.interpolate(repeated_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    with torch.no_grad():
        tensor = model(tensor)
    # deresized_tensor = torch.nn.functional.interpolate(tensor, size=(360, 500), mode='bilinear', align_corners=False)
    derepeated_tensor = tensor.squeeze()
    
    return derepeated_tensor.cpu().numpy()
    


if __name__ == '__main__':
    # model = net()
    # input = torch.randn(1, 1, 360, 500)
    # x = model(input)
    # print(x.shape)
    variabletrain('cuda:1', decompose_mode='retinexd', denoising_mode='0to2')
