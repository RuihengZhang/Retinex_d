import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def resize_and_convert_to_grayscale(directory, size=(500, 360)):

    """
    Resize and convert all .png images in the specified directory to grayscale,
    with a progress bar.
    
    Args:
    - directory (str): The path to the directory containing .png images.
    - size (tuple): The desired size for the resized images, default is (300, 300).
    """
    # Get all .png files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.png')]

    for filename in tqdm(files, desc="Processing images"):
        file_path = os.path.join(directory, filename)
        with Image.open(file_path) as img:
            # Resize and convert to grayscale
            img = img.resize(size).convert("L")
            # Save the image with the same filename
            img.save(file_path)

def save_reflection_and_illumination_log(img_path):

    def single_scale_retinex(img, sigma):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return np.power(10, retinex.astype(np.float32)), cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    
    img = cv2.imread(img_path, 0)  
    sigma = 25
    img = np.where(img==0, 1e-10, img)

    reflection, illumination = single_scale_retinex(img, sigma)
    reflection_normalized = np.uint8(cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX))
    # illumination_normalized = np.uint8(cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX))

    parts = int(img_path.split('/')[-2][-1])

    cv2.imwrite(f'./Images/demo/ref{parts}.png', reflection_normalized)
    cv2.imwrite(f'./Images/demo/ill{parts}.png', illumination)

def decompose_for_folder(folder_path):
    # 该函数用于分解一个文件夹下的图片

    for filename in tqdm(os.listdir(folder_path)):

        part = int(folder_path.split('/')[-1][-1])

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)

            img = cv2.imread(image_path)
            sigma = 25
            img = np.where(img==0, 1e-6, img)

            ill = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
            ill = np.where(ill==0, 1e-6, ill)
            ref = np.power(10, np.log10(img) - np.log10(ill).astype(np.float32))
            ref_norm = np.uint8(cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX))

            cv2.imwrite(f'./Images/S_0123/S_{part}/{filename}', ref_norm)
            cv2.imwrite(f'./Images/E_0123/E_{part}/{filename}', ill)

def reverse_sigle_image(S_path, E_path):
    # 该函数用于将反射图与对应的照射图合成为一张图像

    ref = cv2.imread(S_path, cv2.IMREAD_GRAYSCALE)
    ill = cv2.imread(E_path, cv2.IMREAD_GRAYSCALE)

    ref = np.where(ref==0, 1e-6, ref)
    ill = np.where(ill==0, 1e-6, ill)

    img = ill * ref

    img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

    return img

def decompose(img_path):

    def single_scale_retinex(img, sigma):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return np.power(10, retinex.astype(np.float32)), cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    
    img = cv2.imread(img_path, 0)  
    sigma = 25
    img = np.where(img==0, 1e-10, img)

    reflection, illumination = single_scale_retinex(img, sigma)
    reflection_normalized = np.uint8(cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX))
    img = reflection_normalized * illumination
    img_normed = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

    base_name = os.path.basename(img_path).split('.')[0]
    type_name = int(img_path.split('/')[-2][-1])

    cv2.imwrite(f'./Results/decomposition/ssr/{base_name}_{type_name}_S.png', reflection_normalized)
    cv2.imwrite(f'./Results/decomposition/ssr/{base_name}_{type_name}_E.png', illumination)
    cv2.imwrite(f'./Results/decomposition/ssr/{base_name}_{type_name}_I.png', img_normed)



if __name__ == '__main__':
    
    for i in range(4):
        decompose(f'./Images/I_0123/I_{i}/0.png')