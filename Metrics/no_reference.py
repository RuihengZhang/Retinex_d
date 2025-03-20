from skimage.measure import shannon_entropy
import numpy as np
import cv2
import numpy as np


def local_entropy(image, window_size=3):
    # 获取图像尺寸
    height, width = image.shape[:2]
    entropy_map = np.zeros((height, width))
    # 遍历每个窗口，计算局部熵
    for i in range(height - window_size):
        for j in range(width - window_size):
            patch = image[i:i + window_size, j:j + window_size]
            entropy_map[i, j] = shannon_entropy(patch)
    # 计算平均熵值
    return np.mean(entropy_map)

def edge_density(image):
    image = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image, 100, 200)
    edge_ratio = np.sum(edges) / (image.shape[0] * image.shape[1])

    return edge_ratio

def GSI(image):
    # gradient smoothness index
    image = (image * 255).astype(np.float32)
    grad_x = (np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1)) / 2
    grad_y = (np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)) / 2
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)
    w_mean = 0
    w_std = 0.3
    return w_mean * mean_gradient + w_std * std_gradient







