import cv2
import numpy as np


def non_local_means_denoising(img, h=10, search_window=21, block_size=7):
    
    # 使用 OpenCV 的非局部均值去噪
    img = (img * 255).astype(np.uint8)  # 转换为 uint8 类型，范围 0-255
    denoised_img = cv2.fastNlMeansDenoising(img, None, h, block_size, search_window)
    return denoised_img / 255


