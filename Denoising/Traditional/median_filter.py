import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from Metrics.with_reference import psnr, ssim

def median_filter(img_path, kernel_size=3):
    # 输入待滤波图像的路径，返回去噪后的图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return cv2.medianBlur(img, kernel_size)


def median_filter_with_retinexd(img_path, kernel_size=3):
    # 中值滤波与retinrxd结合，测试效果
    # img_path是待降噪的图像，即I_1, 实际使用时需要读取S_1
    s1_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_1', os.path.basename(img_path))
    e1_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_1', os.path.basename(img_path))
    s1 = cv2.imread(s1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    e1 = cv2.imread(e1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    de_s1 = cv2.medianBlur(s1, kernel_size)
    return e1 * de_s1

def median_filter_with_ssr(img_path):
    def ssr(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.where(img==0, 1e-3, img)
        l = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=50)
        l = np.where(l<=0, 1e-3, l)
        r = np.pow(10, np.log(img)-np.log(l))
        return r, l
    s1, e1 = ssr(img_path)
    de_s1 = cv2.medianBlur(s1, 3)
    return e1 * de_s1

def median_filter_with_msr(img_path):
    def msr(img_path, scales=[7, 15, 31], weights=[0.33, 0.33, 0.33]):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.where(img==0, 1e-3, img)
        r = np.zeros_like(img)
        for scale, weight in zip(scales, weights):
            l = cv2.GaussianBlur(img, ksize=(scale, scale), sigmaX=scale/3)
            l = np.where(l==0, 1e-3, l)
            r += weight * np.pow(10, np.log(img) - np.log(l))
        l = np.pow(10, np.log(img) - np.log(r))
        return r, l
    s1, e1 = msr(img_path)
    de_s1 = cv2.medianBlur(s1, 3)
    return e1 * de_s1

def median_filter_with_wls(img_path):
    def wls(img_path, lambda_val=1.0, alpha=1.2):  # img就是原图的灰度图像(归一化)，输出也是归一化的求解出来的照射图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.where(img==0, 1e-3, img)
        L = np.log1p(np.array(img, dtype=np.float32))
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
        expm1_x = np.expm1(x)
        norm_expm1_x = np.clip(expm1_x, 0, 1)
        l = np.where(norm_expm1_x==0, 1e-3, norm_expm1_x)
        r = np.pow(10, np.log(img)-np.log(l))
        return r, l
    s1, e1 = wls(img_path)
    de_s1 = cv2.medianBlur(s1, 3)
    return e1 * de_s1

def median_filter_with_bf(img_path):
    def bf(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.where(img==0, 1e-3, img)
        flitered_img = cv2.bilateralFilter(img, 9, 75, 75) # 得到双边滤波后的图像
        l = np.clip(flitered_img, 0, 1) # 归一化后作为照射图
        l = np.where(l==0, 1e-3, l)
        r = np.pow(10, np.log(img)-np.log(l)) 
        return r, l
    s1, e1 = bf(img_path)
    de_s1 = cv2.medianBlur(s1, 3)
    return e1 * de_s1

def median_filter_with_gf(img_path):
    def gf(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = np.where(img==0, 1e-3, img)
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
        l = np.where(l==0, 1e-3, l)
        r = np.pow(10, np.log(img)-np.log(l))
        return r, l
    s1, e1 = gf(img_path)
    de_s1 = cv2.medianBlur(s1, 3)
    return e1 * de_s1

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

def caculate_six(original_img_folder_path='Images/I/I_0', noisied_img_folder_path='Images/I/I_1'):
    img_paths = [os.path.join(noisied_img_folder_path, filename) for filename in os.listdir(noisied_img_folder_path)]   # 获取所有的文件路径
    selected_imgs = random.sample(img_paths, 50) # 从中随机取100个
    psnr_array = np.zeros(50)
    ssim_array = np.zeros(50)

    for i, noisy_img_path in enumerate(tqdm(selected_imgs, desc='to be continued...')):
        denoised_img = median_filter_with_retinexd(noisy_img_path)
        original_img = cv2.imread(os.path.join(original_img_folder_path, os.path.basename(noisy_img_path)), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # 计算mse
        psnr_array[i] = psnr(original_img, denoised_img)
        ssim_array[i] = ssim(original_img, denoised_img)

    print(np.mean(psnr_array), np.mean(ssim_array))

def test_one_image():
    noised_img_path = 'Images/I/I_1/229.png'
    groundtruth_path = 'Images/I/I_0/229.png'
    noised_img = cv2.resize(cv2.imread(noised_img_path, cv2.IMREAD_GRAYSCALE), (256, 256))
    groundtruth = cv2.resize(cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE), (256, 256))
    cv2.imwrite('Results/experiment2.2/median_filtering/input.png', noised_img)
    cv2.imwrite('Results/experiment2.2/median_filtering/groundtruth.png', groundtruth)
    # denoised_img = median_filter_with_retinexd(noised_img_path)
    # denoised_img = cv2.resize(denoised_img, (256, 256))
    # cv2.imwrite('Results/experiment2.2/median_filtering/median_filtering_retinexd.png', (denoised_img * 255).astype(np.uint8))




if __name__ == '__main__':
    caculate_six()
    # test_one_image()
    


