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
from torch.autograd import Variable

def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.zeros(N, Cout, Wout, Hout, device=x.device)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features

def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout), device=x.device, dtype=x.dtype)

    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature

class FFDNet(nn.Module):

    def __init__(self, is_gray=True):
        super(FFDNet, self).__init__()

        if is_gray:
            self.num_conv_layers = 15 # all layers number
            self.downsampled_channels = 5 # Conv_Relu in
            self.num_feature_maps = 64 # Conv_Bn_Relu in
            self.output_features = 4 # Conv out
        else:
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.num_feature_maps = 96
            self.output_features = 12
            
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)


    def forward(self, x):
        noise_sigma = torch.full((x.shape[0],), 30 / 255, dtype=torch.float32, device=x.device)
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        x_up = downsample(x.data) # 4 * C * H/2 * W/2
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)
        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = upsample(h_dncnn)

        return y_pred

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
    model = FFDNet()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100 if decompose_mode == 'retinexd' else 25
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
def ffdnet(img, decompose_mode = None):
    model = FFDNet()
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/ffdnet/ffdnet_0to1.pth'))
    else:
        ckt_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/ffdnet/ffdnet_0to1_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 加载图片
    with torch.no_grad():
        denoised_img = model(img)
    return np.array(denoised_img.squeeze())


def rgb_to_gray(rgb_image):
    # 使用加权公式将 RGB 图像转换为灰度图像
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray_image = np.tensordot(rgb_image, weights, axes=([0], [0]))
    return gray_image
# 测试6个指标

def experiment(decompose_mode, denoising_mode, test_mode):

    filenames = args.img_names[2735:]
    groundtruth_folder_path = os.path.join(args.I_folder_path, 'I_'+test_mode[0])
    mse_array, snr_array, ssim_array, ig_array, dpi_array, vif_array = np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50)
    for i, filename in enumerate(tqdm(filenames, desc='正在计算')):
        groundtruth_img_path = os.path.join(groundtruth_folder_path, filename)
        denoised_img = variabletest(noise_filename=filename, decompose_mode=decompose_mode, denoising_mode=denoising_mode, test_mode =test_mode)
        groundtruth = cv2.imread(groundtruth_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if args.img_type == 'RGB':
            denoised_img = rgb_to_gray(denoised_img)
        # 计算mse
        mse_array[i] = np.mean(np.square(denoised_img - groundtruth))
        # 计算snr
        signal_power = np.mean(np.square(groundtruth))
        noise_power = mse_array[i] if mse_array[i] != 0 else 1e-3 # 防止分母为0
        snr_array[i] = 10 * np.log10(signal_power / noise_power)
        # 计算ssim
        if args.img_type == 'L':
            ssim_array[i] = ssim(groundtruth, denoised_img, data_range=1, full=False)
        else:
            ssim_array[i] = ssim(groundtruth, denoised_img, data_range=1, full=False, channel_axis=0)
        # 计算ig
        ig_array[i] = Measure().calculate_entropy(groundtruth) - Measure().calculate_entropy(denoised_img)
        # 计算dpi（基于边缘检测）
        dpi_array[i] = np.mean(np.abs(Measure().edge_detection(groundtruth)-Measure().edge_detection(denoised_img)))
        # 计算vif
        vif_array[i] = vifp(groundtruth, denoised_img)
        
    print(f'mse:{np.mean(mse_array):.4g}, snr:{np.mean(snr_array):.4g}, ssim:{np.mean(ssim_array):.4g}, ig:{np.mean(ig_array):.4g}, dpi:{np.mean(dpi_array):.4g}, vif:{np.mean(vif_array):.4g}')



if __name__ == '__main__':
    # variabletrain(device='cuda:1', decompose_mode='retinexd', denoising_mode='2to3')
    experiment(decompose_mode='retinexd', denoising_mode='2to3', test_mode='2to3')
