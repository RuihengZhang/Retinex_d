import cv2
import numpy as np

def gamma_correction(deg_img, decompose_mode=None):
    inv_gamma = 1.5
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8)
    deg_img = np.array(deg_img * 255, dtype=np.uint8)
    return cv2.LUT(deg_img, table) / 255

