## 该文件下的所有数据集均是我们的实拍
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from Decomposition.traditional import ssr, msr, wls, bf, gf
import numpy as np



class SimpleDataset(Dataset): 
    ## 最简单的数据集
    def __init__(self, enhancement_type):
        self.img_names = os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0')
        self.enhancemnet_type = enhancement_type

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        groundtruth_path = os.path.join(r'Images/I/I_0', img_name)
        
        if self.enhancemnet_type == 'denoising':
        # 去噪任务，使用0和1
            focal_path = os.path.join(r'Images/I/I_1', img_name)
        if self.enhancemnet_type == 'lightening':
            focal_path = os.path.join(r'Images/I/I_2', img_name)

        focal = Image.open(focal_path).convert('RGB')
        groundtruth = Image.open(groundtruth_path).convert('RGB')

        return self.transform(groundtruth), self.transform(focal)
    

class RetinexDDataset(Dataset): 
    ## retinexd数据集
    def __init__(self, enhancement_type):
        self.img_names = os.listdir(r'Images/I/I_0')
        self.enhancemnet_type = enhancement_type

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # 这个数据集最大的不同是focal图像的3个通道是拼接而成的
        groundtruth_path = os.path.join(r'Images/I/I_0', img_name)
        if self.enhancemnet_type == 'denoising':
        # 去噪任务，使用0和1
            focal_1_path = os.path.join(r'Images/I/I_1', img_name)
            focal_2_path = os.path.join(r'Images/S/S_1', img_name)
            focal_3_path = os.path.join(r'Images/E/E_1', img_name)
        if self.enhancemnet_type == 'lightening':
            focal_1_path = os.path.join(r'Images/I/I_2', img_name)
            focal_2_path = os.path.join(r'Images/S/S_2', img_name)
            focal_3_path = os.path.join(r'Images/E/E_2', img_name)

        focal_1 = self.transform(Image.open(focal_1_path).convert('L'))
        focal_2 = self.transform(Image.open(focal_2_path).convert('L'))
        focal_3 = self.transform(Image.open(focal_3_path).convert('L'))

        focal = torch.cat([focal_1, focal_2, focal_3], dim=0)

        groundtruth = Image.open(groundtruth_path).convert('RGB')

        return self.transform(groundtruth), focal
    

class ComplexDataset(Dataset): 
    ## 比较复杂的一个数据集
    def __init__(self, enhancement_type, other_decom):
        self.img_names = os.listdir(r'Images/I/I_0')
        self.enhancemnet_type = enhancement_type
        self.other_decom = other_decom

        self.transform1 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.transform2 = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        groundtruth_path = os.path.join('Images/I/I_0', img_name)
        if self.enhancemnet_type == 'denoising':
            focal_path = os.path.join('Images/I/I_1', img_name)
        if self.enhancemnet_type == 'lightening':
            focal_path = os.path.join('Images/I/I_2', img_name)
        groundtruth = Image.open(groundtruth_path).convert('RGB')
        if self.other_decom == 'ssr':
            focal_1, focal_2, focal_3 = ssr(focal_path)
        if self.other_decom == 'msr':
            focal_1, focal_2, focal_3 = msr(focal_path)
        if self.other_decom == 'wls':
            focal_1, focal_2, focal_3 = wls(focal_path)
        if self.other_decom == 'bf':
            focal_1, focal_2, focal_3 = bf(focal_path)
        if self.other_decom == 'gf':
            focal_1, focal_2, focal_3 = gf(focal_path)
        focal = np.stack((focal_1, focal_2, focal_3), axis=0)
        focal = torch.from_numpy(focal)

        return self.transform1(groundtruth), self.transform2(focal)
    

if __name__ == '__main__':
    dataset = RetinexDDataset('denoising')
    x, y = dataset[0]
    print(x.shape, y.shape, x.max().item(), y.max().item())
        











