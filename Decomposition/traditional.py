import cv2
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

### 该文件实现了5种其它的retinex分解方法，所有函数的输入均为图像的路径，输出为
### 形状为[360, 500]的归一化的numpy数组, 分别为原图像、反射图、照射图

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

