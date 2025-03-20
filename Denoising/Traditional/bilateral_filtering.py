import cv2
import numpy as np


def bilateral_filtering(img):

    img = (img * 255).astype(np.uint8) 
    d = 9                # 滤波器的直径，值越大，效果越强烈
    sigma_color = 75     # 颜色空间的标准差，值越大，颜色相似的像素会更容易混合
    sigma_space = 75     # 坐标空间的标准差，值越大，距离相近的像素会更多混合

    denoised_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    return denoised_img / 255



