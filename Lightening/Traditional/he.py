import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from sewar.full_ref import vifp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import torch.nn.functional as F
# 定义直方图均衡化
def he(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lighten_img = cv2.equalizeHist(img)
    return lighten_img

def he(deg_img, decompose_mode=None):

    lighten_img = cv2.equalizeHist(img)
    return lighten_img


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

def variabletest(I2_path, decompose_mode):
    if decompose_mode == None:
        lighten_I2 = he(I2_path)   # he函数智能对uint8操作
        return lighten_I2.astype(np.float32) / 255.0
    if decompose_mode == 'retinexd':
        S2_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_2', os.path.basename(I2_path))
        E2_path = os.path.join(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_2', os.path.basename(I2_path))
        S2 = cv2.imread(S2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        lighten_E2 = he(E2_path).astype(np.float32) / 255.0
        return S2 * lighten_E2
    else:
        decompose_method = getattr(Decompose(), decompose_mode)
        _, E2, S2 = decompose_method(I2_path)
        E2 = (E2 * 255.0) .astype(np.uint8)
        lighten_E2 = cv2.equalizeHist(E2).astype(np.float32) / 255.0
        return S2 * lighten_E2
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
    caculatesix(decompose_mode='gf')