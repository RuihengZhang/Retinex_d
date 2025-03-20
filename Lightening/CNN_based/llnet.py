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
import torch.nn.functional as F
# 定义LLnet, 实际上是一个自编码器
class LLNet(nn.Module):
    def __init__(self):
        super(LLNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.deconv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.pool1(x2)
        
        x4 = self.relu3(self.conv3(x3))
        x5 = self.relu4(self.conv4(x4))
        x6 = self.pool2(x5)
        
        x7 = self.relu5(self.conv5(x6))
        x8 = self.relu6(self.conv6(x7))
        x9 = self.pool3(x8)
        
        x10 = self.relu7(self.conv7(x9))
        x11 = self.relu8(self.conv8(x10))
        x12 = self.pool4(x11)
        
        # Decoder
        x13 = self.relu9(self.deconv1(x12))
        x14 = self.relu10(self.deconv2(x13))
        x15 = self.relu11(self.deconv3(x14))
        x16 = self.relu12(self.deconv4(x15))
        x17 = self.deconv5(x16)
        
        return x17
# 定义分解法
class Decompose():
    def ssr(self, img_path):
        x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 
        x = np.where(x == 0, 3e-4, x)
        l = cv2.GaussianBlur(x, (7, 7), 50)
        l = np.where(l <= 0, 3e-4, l)
        r = np.power(10, np.log10(x)-np.log10(l)) 
        return x, l, r
    def msr(self, img_path):
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
    def wls(self, img_path):
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
    def bf(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        flitered_img = cv2.bilateralFilter(img, 9, 75, 75) # 得到双边滤波后的图像
        l = np.clip(flitered_img, 0, 1)
        img = np.where(img==0, 3e-4, img)
        l = np.where(l<=0, 3e-4, l)
        r = np.power(10, np.log10(img)-np.log10(l))
        return img, l, r
    def gf(self, img_path):
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
# 定义数据集
class VariableDataset(Dataset): 
    def __init__(self, decompose_mode=None):
        self.img_names = sorted(os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'))[0:2735]
        self.decompose_mode = decompose_mode
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        I2_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_2', img_name)
        I0_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', img_name)
        if self.decompose_mode is None:
            return self.transform(Image.open(I2_img_path).convert('L')), self.transform(Image.open(I0_img_path).convert('L'))   # I2, I0
        if self.decompose_mode == 'retinexd':
            S2_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_2', img_name)
            E2_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_2', img_name)
            return self.transform(Image.open(I2_img_path).convert('L')), self.transform(Image.open(E2_img_path).convert('L')), self.transform(Image.open(S2_img_path).convert('L')), self.transform(Image.open(I0_img_path).convert('L'))   # I2, E2, S2, I0
        else:
            decompose_method = getattr(Decompose(), self.decompose_mode)
            I2, E2, S2 = decompose_method(I2_img_path)
            I2_pil, E2_pil, S2_pil = Image.fromarray((I2 * 255).astype(np.uint8), mode='L'), Image.fromarray((E2 * 255).astype(np.uint8), mode='L'), Image.fromarray((S2 * 255).astype(np.uint8), mode='L')
        return self.transform(I2_pil), self.transform(E2_pil), self.transform(S2_pil), self.transform(Image.open(I0_img_path).convert("L"))   # I2, E2, S2, I0
    
# 定义训练函数
def variabletrain(device, decompose_mode): 
    model = LLNet()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100
    dataset = VariableDataset(decompose_mode=decompose_mode)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        if decompose_mode is None:
            for I2, I0 in progress_bar:
                I2, I0 = I2.to(device), I0.to(device)
                optimizer.zero_grad()
                lighten_I2 = model(I2)
                loss = criterion(lighten_I2, I0)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 10) == 0:
                torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/llnet/llnet_0to2.pth')
        else:
            for I2, E2, S2, I0 in progress_bar:
                I2, E2, S2, I0 = I2.to(device), E2.to(device), S2.to(device), I0.to(device)
                optimizer.zero_grad()
                lighten_I2, lighten_E2 = model(I2), model(E2)
                loss = criterion(S2*lighten_E2, I0) + criterion(lighten_I2, S2*lighten_E2)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 10) == 0:
                save_name = 'llnet_0to2_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/llnet/', save_name))      
# 定义测试函数
def variabletest(I2_path, decompose_mode):
    device = 'cuda:0'
    model = LLNet()
    if decompose_mode == None:
        model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/llnet/llnet_0to2.pth'))
    else:
        ckt_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/llnet/llnet_0to2_with_' + decompose_mode + '.pth'
        model.load_state_dict(torch.load(ckt_path))
    model = model.to(device) # 加载模型
    model.eval()
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    if decompose_mode == None:
        I2 = transform(Image.open(I2_path).convert('L')).to(device).unsqueeze(0) # 加载图片
        with torch.no_grad():
            lighten_img = model(I2)
            lighten_img = F.interpolate(lighten_img, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(lighten_img.cpu())
    if decompose_mode == 'retinexd':
        S2_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_2', os.path.basename(I2_path))
        E2_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_2', os.path.basename(I2_path))
        S2 = transform(Image.open(S2_path).convert('L')).to(device).unsqueeze(0)
        E2 = transform(Image.open(E2_path).convert('L')).to(device).unsqueeze(0)
        with torch.no_grad():
            lighten_img = model(E2) * S2
            lighten_img = F.interpolate(lighten_img, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(lighten_img.cpu())
    else:
        decompose_method = getattr(Decompose(), decompose_mode)
        _, E2, S2 = decompose_method(I2_path)
        E2_pil, S2_pil = Image.fromarray((E2 * 255).astype(np.uint8), mode='L'), Image.fromarray((S2 * 255).astype(np.uint8), mode='L')
        E2, S2 = transform(E2_pil).unsqueeze(0).to(device), transform(S2_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            lighten_img = F.interpolate(model(E2) * S2, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(lighten_img.cpu())
# 定义指标函数
class Measure():
    def calculate_entropy(self, img):
        # 计算直方图，范围为0到1，分为256个bin
        hist = cv2.calcHist([img], [0], None, [256], [0, 1])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 1e-10)  # 防止 log(0)
        entropy = -1.0 * (hist * logs).sum()
        return entropy
    def edge_detection(self, img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        return edge_strength
    
# 测试6个指标
def caculatesix(decompose_mode):
    dark_img_folder_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_2'
    groundtruth_folder_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'
    dark_img_paths = sorted([os.path.join(dark_img_folder_path, filename) for filename in os.listdir(dark_img_folder_path)])[2735:] # 获取要测试的带有噪声的图片
    mse_array = np.zeros(len(dark_img_paths))
    snr_array = np.zeros(len(dark_img_paths))
    ssim_array = np.zeros(len(dark_img_paths))
    ig_array = np.zeros(len(dark_img_paths))
    dpi_array = np.zeros(len(dark_img_paths))
    vif_array = np.zeros(len(dark_img_paths))
    for i, dark_img_path in enumerate(tqdm(dark_img_paths, desc='to be continued...')):
        groundtruth_img_path = os.path.join(groundtruth_folder_path, os.path.basename(dark_img_path))
        lighten_img = variabletest(I2_path=dark_img_path, decompose_mode=decompose_mode)
        groundtruth = cv2.imread(groundtruth_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # 计算mse
        mse_array[i] = np.mean(np.square(lighten_img - groundtruth))
        # 计算snr
        signal_power = np.mean(np.square(groundtruth))
        noise_power = mse_array[i] if mse_array[i] != 0 else 1e-3 # 防止分母为0
        snr_array[i] = 10 * np.log10(signal_power / noise_power)
        # 计算ssim
        ssim_array[i] = ssim(groundtruth, lighten_img, data_range=1, full=False)
        # 计算ig
        ig_array[i] = Measure().calculate_entropy(groundtruth) - Measure().calculate_entropy(lighten_img)
        # 计算dpi（基于边缘检测）
        dpi_array[i] = np.mean(np.abs(Measure().edge_detection(groundtruth)-Measure().edge_detection(lighten_img)))
        # 计算vif
        vif_array[i] = vifp(groundtruth, lighten_img)
        
    print(f'mse:{np.mean(mse_array):.4g}, snr:{np.mean(snr_array):.4g}, ssim:{np.mean(ssim_array):.4g}, ig:{np.mean(ig_array):.4g}, dpi:{np.mean(dpi_array):.4g}, vif:{np.mean(vif_array):.4g}')
    


if __name__ == '__main__':
    # variabletrain(device='cuda:1', decompose_mode='gf')
    # caculatesix(decompose_mode='gf')
    model = LLNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
