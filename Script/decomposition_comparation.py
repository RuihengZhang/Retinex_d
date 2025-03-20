# import os
# print(os.getcwd())
# import sys
# print(sys.path)

### 该文件是分解方法对比实验中的实验1，只比较不同的分解方法
from Decomposition.traditional import ssr, msr, wls, bf, gf
from Metrics.with_reference import ssim
from Metrics.no_reference import local_entropy, edge_density, GSI
import numpy as np
from PIL import Image
import cv2
import random


def decompose(method, I_path):

    if method == "ssr":
        return ssr(I_path)
    if method =='msr':
        return msr(I_path)
    if method =='wls':
        return wls(I_path)
    if method =='bf':
        return bf(I_path)
    if method =='gf':
        return gf(I_path)
    
def load(image_path):

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is None:
        raise ValueError(f"无法读取图像，请检查路径: {image_path}")
    
    # 转换为浮点型并归一化到 [0, 1]
    normalized_image = gray_image.astype(np.float32) / 255.0
    
    return normalized_image
    
def test_one_image(method):
    # 测试各种分解方法并保存结果，就测一张图片

    I0_path = 'Images/I/I_0/18.png'
    I1_path = 'Images/I/I_1/18.png'
    I2_path = 'Images/I/I_2/18.png'

    S0_path = 'Images/S/S_0/18.png'
    S1_path = 'Images/S/S_1/18.png'
    S2_path = 'Images/S/S_2/18.png'

    E0_path = 'Images/E/E_0/18.png'
    E1_path = 'Images/E/E_1/18.png'
    E2_path = 'Images/E/E_2/18.png'

    if method != 'retinexd':
        x0, l0, r0 = decompose(method, I0_path)
        x1, l1, r1 = decompose(method, I1_path)
        x2, l2, r2 = decompose(method, I2_path)

    if method == 'retinexd':
        x0, l0, r0 = load(I0_path), load(E0_path), load(S0_path)
        x1, l1, r1 = load(I1_path), load(E1_path), load(S1_path)
        x2, l2, r2 = load(I2_path), load(E2_path), load(S2_path)

    E0 = (l0 * 255).astype(np.uint8)
    S0 = (r0 * 255).astype(np.uint8)
    E1 = (l1 * 255).astype(np.uint8)
    S1 = (r1 * 255).astype(np.uint8)
    E2 = (l2 * 255).astype(np.uint8)
    S2 = (r2 * 255).astype(np.uint8)

    E0_image = Image.fromarray(E0, mode='L')  
    E0_image.save(f'Results/experiment2.1/{method}_E0.png')
    S0_image = Image.fromarray(S0, mode='L')
    S0_image.save(f'Results/experiment2.1/{method}_S0.png')
    E1_image = Image.fromarray(E1, mode='L')
    E1_image.save(f'Results/experiment2.1/{method}_E1.png')
    S1_image = Image.fromarray(S1, mode='L')
    S1_image.save(f'Results/experiment2.1/{method}_S1.png')
    E2_image = Image.fromarray(E2, mode='L')
    E2_image.save(f'Results/experiment2.1/{method}_E2.png')
    S2_image = Image.fromarray(S2, mode='L')
    S2_image.save(f'Results/experiment2.1/{method}_S2.png')

def measure(method):
    # 测试照射图的平滑程度, 直接出比较结果

    num = random.randint(1, 100)

    I0_path = f'Images/I/I_0/{num}.png'
    I1_path = f'Images/I/I_1/{num}.png'
    I2_path = f'Images/I/I_2/{num}.png'
    E0_path = f'Images/E/E_0/{num}.png'
    E1_path = f'Images/E/E_1/{num}.png'
    E2_path = f'Images/E/E_2/{num}.png'
    S0_path = f'Images/S/S_0/{num}.png'
    S1_path = f'Images/S/S_1/{num}.png'
    S2_path = f'Images/S/S_2/{num}.png'

    if method != 'retinexd':
        x0, l0, r0 = decompose(method, I0_path)
        x1, l1, r1 = decompose(method, I1_path)
        x2, l2, r2 = decompose(method, I2_path)

    if method == 'retinexd':
        x0, l0, r0 = load(I0_path), load(E0_path), load(S0_path)
        x1, l1, r1 = load(I1_path), load(E1_path), load(S1_path)
        x2, l2, r2 = load(I2_path), load(E2_path), load(S2_path)

    print(GSI(r0), GSI(r1), GSI(r2))

    

    


if __name__ == '__main__':

    
    
    method = 'ssr'
    measure(method)
    method = 'msr'
    measure(method)
    method = 'wls'
    measure(method)
    method = 'bf'
    measure(method)
    method = 'gf'
    measure(method)
    method = 'retinexd'
    measure(method)

    

    

    



    

    
    





