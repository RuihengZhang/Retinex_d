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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
# 几种分解方法
def ssr(img_path):
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 
    x = np.where(x == 0, 3e-4, x)
    l = cv2.GaussianBlur(x, (7, 7), 50)
    l = np.where(l <= 0, 3e-4, l)
    r = np.power(10, np.log10(x)-np.log10(l)) 
    return x, l, r
def msr(img_path):
    scales=[7, 15, 31]
    weights = [1.0 / len(scales)] * len(scales)
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    x = np.where(x == 0, 3e-4, x)
    r = np.zeros_like(x)
    for scale, weight in zip(scales, weights):
        l = cv2.GaussianBlur(x, (scale, scale), scale/3)
        l = np.where(l <= 0, 3e-4, l)
        r += weight * np.power(10, np.log10(x) - np.log10(l))
    r = np.where(r <= 0, 3e-4, r)
    l = np.power(10, np.log10(x) - np.log10(r))
    return x, l, r
def wls(img_path):
    def wls_filter(img, lambda_val=1.0, alpha=1.2): 
        L = img
        small_num = 1e-4
        height, width = L.shape
        k = height * width
        dy = np.diff(L, 1, 0)
        dy = -lambda_val / (np.abs(dy)**alpha + small_num)
        dy = np.pad(dy, ((0, 1), (0, 0)), 'constant')
        dx = np.diff(L, 1, 1)
        dx = -lambda_val / (np.abs(dx)**alpha + small_num)
        dx = np.pad(dx, ((0, 0), (0, 1)), 'constant')
        B = np.zeros_like(L)
        B[1:, :] = dy[:-1, :]
        B[:, 1:] += dx[:, :-1]
        A_diag = 1 - B
        A_off_diag = np.pad(B[:-1, :], ((1, 0), (0, 0)), 'constant') + np.pad(B[:, :-1], ((0, 0), (1, 0)), 'constant')
        A = diags([A_diag.flatten(), A_off_diag.flatten(), A_off_diag.flatten()], [0, -1, 1], shape=(k, k)).tocsc()
        L = L.flatten()
        x = spsolve(A, L)
        x = x.reshape((height, width))
        x = np.clip(x, 0, 1)
        return x
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = np.where(img <= 0, 3e-4, img)
    L = wls_filter(img)  # 输入的img就已经是一张灰度图像了,已经归一化了，得到的为一张归一化的照射图
    L = np.where(L <= 0, 3e-4, L)
    R = np.power(10, np.log10(img)-np.log10(L))
    l, r = L.squeeze(), R.squeeze()
    return np.array(img), l, r
def bf(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    flitered_img = cv2.bilateralFilter(img, 9, 75, 75) # 得到双边滤波后的图像
    l = np.clip(flitered_img, 0, 1)
    img = np.where(img==0, 3e-4, img)
    l = np.where(l<=0, 3e-4, l)
    r = np.power(10, np.log10(img)-np.log10(l))
    return img, l, r
def gf(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    p = img # 引导图像
    r = 9  # 滤波半径
    eps = 0.01  # 正则化参数
    # 应用引导滤波
    mean_I = cv2.boxFilter(img, -1, (r,r))
    mean_p = cv2.boxFilter(p, -1, (r,r))
    mean_Ip = cv2.boxFilter(img * p, -1, (r,r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(img * img, -1, (r,r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r,r))
    mean_b = cv2.boxFilter(b, -1, (r,r))
    l = mean_a * img + mean_b # 得到照射图
    l = np.clip(l, 0, 1)
    img = np.where(img == 0, 3e-4, img)
    l = np.where(l <= 0, 3e-4, l)
    r = np.power(10, np.log10(img)-np.log10(l))
    return img, l, r
# 模型与数据集的定义
class DnCNN(nn.Module):  # 定义去噪模型
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out  
class I0toI1(Dataset): # 第一个数据集，I1作为带噪声的图片，I0作为groundtruth
    def __init__(self, noisy_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1', gt_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', transform=transforms.Compose([transforms.ToTensor(),])):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))[0:2735]
        self.gt_images = sorted(os.listdir(gt_dir))[0:2735]  # 排序
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])
        noisy_image = Image.open(noisy_image_path).convert("L")  
        gt_image = Image.open(gt_image_path).convert("L")   
        noisy_image = self.transform(noisy_image)
        gt_image = self.transform(gt_image)

        return noisy_image, gt_image # 先噪声后标签
class I0I1S0S1E1(Dataset): # 第二个数据集
    def __init__(self, I0_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', I1_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1', S0_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_0', S1_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_1', E1_dir='/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_1', transform=transforms.Compose([transforms.ToTensor(),])):
        self.img_paths = sorted(os.listdir(I0_dir))[0:2735]
        self.transform = transform    
        self.I0_dir = I0_dir
        self.I1_dir = I1_dir
        self.S0_dir = S0_dir
        self.S1_dir = S1_dir
        self.E1_dir = E1_dir
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        I0 = Image.open(os.path.join(self.I0_dir, self.img_paths[idx])).convert("L")  
        I1 = Image.open(os.path.join(self.I1_dir, self.img_paths[idx])).convert("L") 
        S0 = Image.open(os.path.join(self.S0_dir, self.img_paths[idx])).convert("L") 
        S1 = Image.open(os.path.join(self.S1_dir, self.img_paths[idx])).convert("L") 
        E1 = Image.open(os.path.join(self.E1_dir, self.img_paths[idx])).convert("L") 
        return self.transform(I0), self.transform(I1), self.transform(S0), self.transform(S1), self.transform(E1)
class VariableDataset(Dataset): 
    def __init__(self, decompose_method):
        self.img_names = os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0')
        self.decompose_method = decompose_method
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        I1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1', img_name)
        I0_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', img_name)
        I1, E1, S1 = self.decompose_method(I1_img_path)
        I0 = cv2.imread(I0_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return torch.Tensor(I1).unsqueeze(0), torch.Tensor(E1).unsqueeze(0), torch.Tensor(S1).unsqueeze(0), torch.Tensor(I0).unsqueeze(0)
    
# 评价指标计算函数的定义
def calculate_entropy(img):
    """
    计算值域范围为0到1的灰度图像的熵
    """
    # 计算直方图，范围为0到1，分为256个bin
    hist = cv2.calcHist([img], [0], None, [256], [0, 1])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-10)  # 防止 log(0)
    entropy = -1.0 * (hist * logs).sum()
    return entropy
def edge_detection(img):
    """
    使用Sobel算子进行边缘检测
    """
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    return edge_strength
# 训练函数
def train1(): # 第一个训练, 使用数据集为I0和I1(仅测试去噪能力)
    device = 'cuda:0'
    model = DnCNN()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = I0toI1()
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for noisy_image, groundtruth in progress_bar:
            noisy_image, groundtruth = noisy_image.to(device), groundtruth.to(device)
            optimizer.zero_grad()
            denoised_image = model(noisy_image)
            loss = criterion(denoised_image, groundtruth)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch+1 == epochs or epoch == 0:
            torch.save(model.module.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1.pth')

def train2(): # 第二个训练, 将DnCnn结合RetinexD方法使用, 涉及数据集为I0和I1、S0和S1
    device = 'cuda:3'
    model = DnCNN()
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[2, 3])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = I0I1S0S1E1()
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for I0, I1, S0, S1, E1 in progress_bar:
            I0, I1, S0, S1, E1 = I0.to(device), I1.to(device), S0.to(device), S1.to(device), E1.to(device)
            optimizer.zero_grad()
            denoised_I1 = model(I1)
            denoised_S1 = model(S1)
            loss =  0 * criterion(E1*denoised_S1, I0) + criterion(denoised_I1, E1*denoised_S1)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch+1 == epochs or epoch == 0:
            torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1_with_retinexd_g.pth')
def train3(): # 将DnCnn结合其它分解方法使用
    device = 'cuda:2'                             # 更换显卡
    model = DnCNN()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = VariableDataset(decompose_method=gf)    # 更换数据集
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for I1, E1, S1, I0 in progress_bar:
            I1, E1, S1, I0 = I1.to(device), E1.to(device), S1.to(device), I0.to(device)
            optimizer.zero_grad()
            denoised_I1, denoised_S1 = model(I1), model(S1)
            loss = criterion(E1*denoised_S1, I0) + criterion(denoised_I1, E1*denoised_S1)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch+1 == epochs or epoch == 0:
            torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1_with_gf.pth')      # 更换存储位置
# 测试函数
def dncnn_denoised(noisy_img_path):
    device = 'cuda:0'
    model = DnCNN()
    model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1.pth'))
    model = model.to(device) # 加载模型
    model.eval()
    noisy_img = transforms.ToTensor()(Image.open(noisy_img_path).convert('L')).to(device).unsqueeze(0) # 加载图片
    with torch.no_grad():
        denoised_img = model(noisy_img).squeeze()
    return denoised_img.cpu()
def dncnn_with_retinexd(I1_img_path):
    device = 'cuda:0'
    model = DnCNN()
    model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1_with_retinexd_g.pth'))
    model = model.to(device) # 加载模型
    model.eval()
    S1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_1', os.path.basename(I1_img_path))
    E1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_1', os.path.basename(I1_img_path))
    S1 = transforms.ToTensor()(Image.open(S1_img_path).convert('L')).to(device).unsqueeze(0) 
    E1 = transforms.ToTensor()(Image.open(E1_img_path).convert('L')).to(device).unsqueeze(0)
    S1, E1 = S1.to(device), E1.to(device)
    with torch.no_grad():
        denoised_img = (model(S1)*E1).squeeze()
    return denoised_img.cpu()

def dncnn_with_others(I1_img_path, decompose_method=gf):
    device = 'cuda:0'
    model = DnCNN()
    model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/dncnn/0to1_with_gf.pth'))
    model = model.to(device) # 加载模型
    model.eval()
    _, E1, S1 = decompose_method(I1_img_path)
    S1, E1 = torch.Tensor(S1).unsqueeze(0).unsqueeze(0).to(device), torch.Tensor(E1).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_img = (model(S1)*E1).squeeze()
    return np.array(denoised_img.cpu())
    
def caculate_six(noisied_img_folder_path, groundtruth_folder_path):
    noisied_img_paths = sorted([os.path.join(noisied_img_folder_path, filename) for filename in os.listdir(noisied_img_folder_path)])[2735:] # 获取要测试的带有噪声的图片
    mse_array = np.zeros(len(noisied_img_paths))
    snr_array = np.zeros(len(noisied_img_paths))
    ssim_array = np.zeros(len(noisied_img_paths))
    ig_array = np.zeros(len(noisied_img_paths))
    dpi_array = np.zeros(len(noisied_img_paths))
    vif_array = np.zeros(len(noisied_img_paths))
    for i, noisy_img_path in enumerate(tqdm(noisied_img_paths, desc='to be continued...')):
        groundtruth_img_path = os.path.join(groundtruth_folder_path, os.path.basename(noisy_img_path))
        denoised_img = dncnn_with_others(noisy_img_path)
        groundtruth = cv2.imread(groundtruth_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # 计算mse
        mse_array[i] = np.mean(np.square(denoised_img - groundtruth))
        # 计算snr
        signal_power = np.mean(np.square(groundtruth))
        noise_power = mse_array[i] if mse_array[i] != 0 else 1e-3 # 防止分母为0
        snr_array[i] = 10 * np.log10(signal_power / noise_power)
        # 计算ssim
        ssim_array[i] = ssim(groundtruth, denoised_img, data_range=1, full=False)
        # 计算ig
        ig_array[i] = calculate_entropy(groundtruth) - calculate_entropy(denoised_img)
        # 计算dpi（基于边缘检测）
        dpi_array[i] = np.mean(np.abs(edge_detection(groundtruth)-edge_detection(denoised_img)))
        # 计算vif
        vif_array[i] = vifp(groundtruth, denoised_img)
        
    print(f'mse:{np.mean(mse_array):.4g}, snr:{np.mean(snr_array):.4g}, ssim:{np.mean(ssim_array):.4g}, ig:{np.mean(ig_array):.4g}, dpi:{np.mean(dpi_array):.4g}, vif:{np.mean(vif_array):.4g}')

if __name__ == '__main__':
    # train2()
    # caculate_six(noisied_img_folder_path='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1', groundtruth_folder_path='/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0')
    # train3()
    model = DnCNN()
    x = torch.randn(1, 3, 360, 500)
    y = model(x)
    print(y.shape)

