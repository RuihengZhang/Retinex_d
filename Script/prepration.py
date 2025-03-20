import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def add_gaussian_noise(image, mean=0, std=25):
    # 添加高斯噪声
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # 确保像素值在 [0, 255] 范围内
    return noisy_image
def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    # 添加椒盐噪声
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image
def add_poisson_noise(image):
    # 添加泊松噪声
    noisy_image = np.random.poisson(image).astype(np.float32)
    noisy_image = np.clip(noisy_image, 0, 255)  # 确保像素值在 [0, 255] 范围内
    return noisy_image
def degration(input_dir, output_dir):
    # 获取一批图像，选择输入路径与输出的路径，按照一定的比例将输入文件夹的图片添加不同类型的噪声，并存到输出路径
    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    input_files.sort()
    for file_path in input_files[0:1000]:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = add_gaussian_noise(image)
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, noisy_image)
    for file_path in input_files[1000:2000]:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = add_salt_and_pepper_noise(image)
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, noisy_image)
    for file_path in input_files[2000:]:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = add_poisson_noise(image)
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, noisy_image)

def adjust_contrast(image, alpha=0.5):
    """
    调整图像的对比度
    :param image: 输入图像 (NumPy 数组)
    :param alpha: 对比度系数，值越接近 0，对比度越低，值越接近 1，对比度越高
    :return: 调整对比度后的图像 (NumPy 数组)
    """
    image = image.astype(np.float32)
    mean = np.mean(image)
    adjusted = alpha * image + (1 - alpha) * mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

def degration_contrast(input_dir, output_dir):
    # 获取一批图像,按照相同的方式降低其对比度（随机降的程度）
    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    input_files.sort()
    for file_path in input_files[0:50]:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = adjust_contrast(image, random.uniform(0.05, 0.1))
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, noisy_image)
    for file_path in input_files[50:]:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = adjust_contrast(image, random.uniform(0.1, 0.5))
        save_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, noisy_image)



if __name__ == '__main__':
    # degration('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I_0', '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I_1')  # 给图像添加噪声
    degration_contrast('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I_0', '/mnt/jixie8t/zd_new/Code/RetinexD/Images/I_2')