import numpy as np
from skimage.metrics import structural_similarity
### 实现了一些指标测试的方法，主要是一些需要参考图像的指标，输入为numpy数组

def psnr(ref_img, img):

        mse = np.mean((ref_img - img) ** 2)
        psnr = 20 * np.log10(1 / np.sqrt(mse))

        return psnr

def ssim(ref_img, img):

        ssim = structural_similarity(ref_img, img, data_range=1, full=False)

        return ssim