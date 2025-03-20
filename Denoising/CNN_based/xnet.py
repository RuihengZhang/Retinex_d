import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import argparse
import random
from tqdm import tqdm
import cv2
import numpy as np

class Xnet(nn.Module):
  def __init__(self):
    super(Xnet,self).__init__() 
    self.conv1 = nn.Conv2d(1,64,kernel_size = 3, padding = 1)                     #1 x 256 x 256 > 64 x 256 x 256
    self.batch_norm1 = nn.BatchNorm2d(64)                                         #BN1 = 64                                                
    self.pool = nn.MaxPool2d(2,2)                                                 # 64 x 128 x 128 
    self.conv2 = nn.Conv2d(64,128, kernel_size =3, padding =1)                    # 128 x 128 x 128
    self.batch_norm2 = nn.BatchNorm2d(128)                                        # bn2 = 128
    self.conv3 = nn.Conv2d(128,256, kernel_size = 3, padding = 1)                 # 256 x 128 x 128
    self.batch_norm3 = nn.BatchNorm2d(256) 
    self.conv4 = nn.Conv2d(256,512, kernel_size = 3, padding = 1)
    self.batch_norm4 = nn.BatchNorm2d(512) 
    self.conv5 = nn.Conv2d(512,512, kernel_size = 3, padding = 1)
    self.conv6 = nn.Conv2d(512,256, kernel_size = 3, padding = 1)
    self.conv7 = nn.Conv2d(256,256, kernel_size = 3, padding = 1)
    self.conv8 = nn.Conv2d(256,128,kernel_size = 3, padding = 1)
    self.conv9 = nn.Conv2d(128,256,kernel_size = 3, padding = 1)
    self.conv10 = nn.Conv2d(256, 512, kernel_size = 3, padding =1)
    self.conv11 = nn.Conv2d(512,512, kernel_size = 3, padding = 1)
    self.conv12 = nn.Conv2d(512,256, kernel_size = 3, padding = 1)
    self.conv13 = nn.Conv2d(256,128, kernel_size = 3, padding = 1)
    self.conv14 = nn.Conv2d(128,64, kernel_size =3, padding = 1)
    self.conv15 = nn.Conv2d(64, 1, kernel_size = 1)
    self.fc1 = nn.Linear(1*256*256, 64)
    self.fc2 = nn.Linear(64, 1)
    
  def forward(self, x):
    act1 = F.relu(self.batch_norm1(self.conv1(x)))          #64 x 256 x 256
    x = self.pool(act1)                                     #64 x 128 x 128
    act2 = F.relu(self.batch_norm2(self.conv2(x)))          #128 x 128 x 128
    x = self.pool(act2)                                     #128 x 64 x 64
    act_3 = F.relu(self.batch_norm3(self.conv3(x)))         #256 x 64 x 64
    x = self.pool(act_3)                                    #256 x 32 x 32
    x = F.relu(self.batch_norm4(self.conv4(x)))   #512 x 32 x 32
    x = F.relu(self.batch_norm4(self.conv5(x)))   #512 x 32 x 32
    x = F.upsample(x, size = 64)   #6
    x = F.relu(self.batch_norm3(self.conv6(x)))   #256 x 64 x 64    
    x = x.add(act_3)    
    x = F.upsample(x, size = 128)                 #256 x 128 x 128
    x = F.relu(self.batch_norm3(self.conv7(x)))   #256 x 128 x 128
    # x = x.add(act2)    
    act_8 = F.relu(self.batch_norm2(self.conv8(x))) #128 x 128 x 128
    x = self.pool(act_8)                            #128 x 64 x 64
    act_9 = F.relu(self.batch_norm3(self.conv9(x))) #256 x 64 x 64
    x = self.pool(act_9)                            #256 x 32 x 32
    x = F.relu(self.batch_norm4(self.conv10(x)))    #512 x 32 x 32
    x = F.relu(self.batch_norm4(self.conv11(x)))    #512 x 32 x 32    
    x = F.upsample(x, size = 64)                    #512 x 64 x 64
    x = F.relu(self.batch_norm3(self.conv12(x)))    #256 x 64 x 64
    x = x.add(act_9)    
    x = F.upsample(x, size = 128)                   #256 x 128 x 128 
    x = F.relu(self.batch_norm2(self.conv13(x)))    #128 x 128 x 128
    x = x.add(act_8)    
    x = F.upsample(x, size = 256)                   #128 x 256 x 256
    x = F.relu(self.batch_norm1(self.conv14(x)))    #64 x 256 x 256
    x = x.add(act1)
    x = self.conv15(x)                              #13 x 256 x 256
    # x = x.view(-1,1*256*256)
    # x = F.relu(self.fc1(x))
    # x = nn.Sigmoid()(self.fc2(x))
    return x
    
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
    model = Xnet()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60 if decompose_mode == 'retinexd' else 40
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
            if epoch+1 == epochs or (epoch % 5) == 0:
                save_name = f'{args.py_filename}_{denoising_mode}_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/{args.py_filename}/', save_name))

def xnet(img, decompose_mode=None):
    model = Xnet()
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/xnet/xnet_0to1.pth'))
    else:
        ckt_path = f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/xnet/xnet_0to1_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    input_array = img
    input_array_resized = cv2.resize(input_array, (256, 256))
    tensor = torch.tensor(input_array_resized, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
    with torch.no_grad():
        tensor = model(tensor)
    tensor = tensor.squeeze()
    numpy_array = tensor.numpy()
    numpy_resized = cv2.resize(numpy_array, (500, 360))
  # 输出: (360, 500)
    return numpy_resized

if __name__ == '__main__':
    # model = Xnet()
    # y = model(torch.randn(1, 1, 256, 256))
    # print(y.shape)
    variabletrain('cuda:1', 'retinexd', '0to1')