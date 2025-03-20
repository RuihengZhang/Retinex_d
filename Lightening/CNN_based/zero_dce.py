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

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)

		return enhance_image
	

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
    model = enhance_net_nopool()      # 为了尽可能快的训练，使用单卡训练
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

def zero_dce(img, decompose_mode=None):
    model = enhance_net_nopool()
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/zero_dce/zero_dce_0to2.pth'))
    else:
        ckt_path = f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/zero_dce/zero_dce_0to2_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    input_array = img
    tensor = torch.tensor(input_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    repeated_tensor = tensor.repeat(1, 3, 1, 1)

    with torch.no_grad():
        tensor = model(repeated_tensor)
    tensor = tensor.squeeze()
    numpy_array = tensor[0].numpy()

    return numpy_array

if __name__ == '__main__':
	# model = enhance_net_nopool()
	# x = torch.randn(1, 3, 360, 500)  # output:[1, 3, 360, 500]
	# with torch.no_grad():
	# 	y = model(x)
	# print(y.shape)
    variabletrain('cuda:0', None, '0to2')





