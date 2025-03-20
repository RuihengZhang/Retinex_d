import pywt
import numpy as np
import cv2

def wavelet_denoising(img, wavelet='db1', level=2, threshold=20):
    # 将图像转换为浮点数
    img = (img * 255).astype(np.float32)
    # 小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    # 对高频系数进行阈值处理
    thresholded_coeffs = [coeffs[0]]  # 保留低频系数
    for detail_coeffs in coeffs[1:]:
        thresholded_detail_coeffs = tuple(np.where(np.abs(detail) > threshold, detail, 0) for detail in detail_coeffs)
        thresholded_coeffs.append(thresholded_detail_coeffs)
    
    # 小波重构
    denoised_img = pywt.waverec2(thresholded_coeffs, wavelet)

    return denoised_img / 255




