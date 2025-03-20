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
# 定义Unet
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x
class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)

        return d1
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
        I1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1', img_name)
        I0_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', img_name)
        if self.decompose_mode is None:
            return self.transform(Image.open(I1_img_path).convert('L')), self.transform(Image.open(I0_img_path).convert('L'))   # I1, I0
        if self.decompose_mode == 'retinexd':
            S1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_1', img_name)
            E1_img_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_1', img_name)
            return self.transform(Image.open(I1_img_path).convert('L')), self.transform(Image.open(E1_img_path).convert('L')), self.transform(Image.open(S1_img_path).convert('L')), self.transform(Image.open(I0_img_path).convert('L'))   # I1, E1, S1, I0
        else:
            decompose_method = getattr(Decompose(), self.decompose_mode)
            I1, E1, S1 = decompose_method(I1_img_path)
            I1_pil, E1_pil, S1_pil = Image.fromarray((I1 * 255).astype(np.uint8), mode='L'), Image.fromarray((E1 * 255).astype(np.uint8), mode='L'), Image.fromarray((S1 * 255).astype(np.uint8), mode='L')
        return self.transform(I1_pil), self.transform(E1_pil), self.transform(S1_pil), self.transform(Image.open(I0_img_path).convert("L"))   # I1, E1, S1, I0
# 定义训练函数
def variabletrain(device, decompose_mode): 
    model = U_Net()      # 为了尽可能快的训练，使用单卡训练
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
            for I1, I0 in progress_bar:
                I1, I0 = I1.to(device), I0.to(device)
                optimizer.zero_grad()
                denoised_I1 = model(I1)
                loss = criterion(denoised_I1, I0)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or epoch == 0:
                torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/unet/unet_0to1.pth')
        else:
            for I1, E1, S1, I0 in progress_bar:
                I1, E1, S1, I0 = I1.to(device), E1.to(device), S1.to(device), I0.to(device)
                optimizer.zero_grad()
                denoised_I1, denoised_S1 = model(I1), model(S1)
                loss = criterion(E1*denoised_S1, I0) + criterion(denoised_I1, E1*denoised_S1)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or epoch == 0:
                save_name = 'unet_0to1_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/unet/', save_name))      
# 定义测试函数
def variabletest(I1_path, decompose_mode):
    device = 'cuda:0'
    model = U_Net()
    if decompose_mode == None:
        model.load_state_dict(torch.load(r'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/unet/unet_0to1.pth'))
    else:
        ckt_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/denoising/unet/unet_0to1_with_' + decompose_mode + '.pth'
        model.load_state_dict(torch.load(ckt_path))
    model = model.to(device) # 加载模型
    model.eval()
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    if decompose_mode == None:
        I1 = transform(Image.open(I1_path).convert('L')).to(device).unsqueeze(0) # 加载图片
        with torch.no_grad():
            denoised_img = model(I1)
            denoised_img = F.interpolate(denoised_img, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(denoised_img.cpu())
    if decompose_mode == 'retinexd':
        S1_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_1', os.path.basename(I1_path))
        E1_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_1', os.path.basename(I1_path))
        S1 = transform(Image.open(S1_path).convert('L')).to(device).unsqueeze(0)
        E1 = transform(Image.open(E1_path).convert('L')).to(device).unsqueeze(0)
        with torch.no_grad():
            denoised_img = model(S1) * E1
            denoised_img = F.interpolate(denoised_img, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(denoised_img.cpu())
    else:
        decompose_method = getattr(Decompose(), decompose_mode)
        _, E1, S1 = decompose_method(I1_path)
        E1_pil, S1_pil = Image.fromarray((E1 * 255).astype(np.uint8), mode='L'), Image.fromarray((S1 * 255).astype(np.uint8), mode='L')
        E1, S1 = transform(E1_pil).unsqueeze(0).to(device), transform(S1_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_img = F.interpolate(model(S1) * E1, size=(360, 500), mode='bilinear', align_corners=False).squeeze()
        return np.array(denoised_img.cpu())
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
    noisied_img_folder_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1'
    groundtruth_folder_path = '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'
    noisied_img_paths = sorted([os.path.join(noisied_img_folder_path, filename) for filename in os.listdir(noisied_img_folder_path)])[2735:] # 获取要测试的带有噪声的图片
    mse_array = np.zeros(len(noisied_img_paths))
    snr_array = np.zeros(len(noisied_img_paths))
    ssim_array = np.zeros(len(noisied_img_paths))
    ig_array = np.zeros(len(noisied_img_paths))
    dpi_array = np.zeros(len(noisied_img_paths))
    vif_array = np.zeros(len(noisied_img_paths))
    for i, noisy_img_path in enumerate(tqdm(noisied_img_paths, desc='to be continued...')):
        groundtruth_img_path = os.path.join(groundtruth_folder_path, os.path.basename(noisy_img_path))
        denoised_img = variabletest(I1_path=noisy_img_path, decompose_mode=decompose_mode)
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
        ig_array[i] = Measure().calculate_entropy(groundtruth) - Measure().calculate_entropy(denoised_img)
        # 计算dpi（基于边缘检测）
        dpi_array[i] = np.mean(np.abs(Measure().edge_detection(groundtruth)-Measure().edge_detection(denoised_img)))
        # 计算vif
        vif_array[i] = vifp(groundtruth, denoised_img)
        
    print(f'mse:{np.mean(mse_array):.4g}, snr:{np.mean(snr_array):.4g}, ssim:{np.mean(ssim_array):.4g}, ig:{np.mean(ig_array):.4g}, dpi:{np.mean(dpi_array):.4g}, vif:{np.mean(vif_array):.4g}')
    


if __name__ == '__main__':
    # variabletrain(device='cuda:1', decompose_mode='gf')
    # caculatesix(decompose_mode='gf')
    model = U_Net()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)