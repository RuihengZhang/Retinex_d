import cv2
import numpy as np

def enhance_brightness_gray(deg_img, decompose_mode=None):
    deg_img = np.array(deg_img * 255, dtype=np.uint8)
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # 对灰度图像应用 CLAHE
    enhanced_image = clahe.apply(deg_img)

    return enhanced_image / 255




