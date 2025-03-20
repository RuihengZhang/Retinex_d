import cv2
import numpy as np

def gaussian_filtering(image):
    # 输入为归一化的数组
    kernel_size = (5, 5)
    sigma = 1.0
    gaussian_blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # gaussian_blurred_uint8 = np.clip(gaussian_blurred * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite('/mnt/jixie8t/zd_new/Code/RetinexD/Demo/pre/gaussian_filtering.png', gaussian_blurred_uint8)

    return gaussian_blurred


if __name__ == '__main__':
    gaussian_filtering('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_1/612.png')





