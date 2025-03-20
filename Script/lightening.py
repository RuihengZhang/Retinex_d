import sys
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from Lightening.Traditional.gamma_correction import gamma_correction as lightening

class Metrics():
    
    def psnr(self, ref_img, img):

        mse = np.mean((ref_img - img) ** 2)
        psnr = 20 * np.log10(1 / np.sqrt(mse))

        return psnr
    
    def ssim(self, ref_img, img):

        ssim = structural_similarity(ref_img, img, data_range=1, full=False)

        return ssim
    
    def ig(self, ref_img, img):

        ref_img, img = np.clip(ref_img, 0, 1), np.clip(img, 0, 1)
        ref_img, img = (ref_img * 255).astype(np.uint8), (img * 255).astype(np.uint8)

        ref_img_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 256])
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        ref_img_hist = ref_img_hist.ravel() / ref_img_hist.sum()
        img_hist = img_hist.ravel() / img_hist.sum()

        ref_img_hist = ref_img_hist[ref_img_hist > 0]
        img_hist = img_hist[img_hist > 0]

        ref_img_entropy = -np.sum(ref_img_hist * np.log2(ref_img_hist))
        img_entropy = -np.sum(img_hist * np.log2(img_hist))

        return ref_img_entropy - img_entropy
    
    def dpi(self, ref_img, img):
        
        def edge_stren(img):
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.sqrt(grad_x**2 + grad_y**2)
            return edge_strength
        
        return np.mean(edge_stren(img) / (edge_stren(ref_img) + 3e-2))
    
    def vif(self, ref_img, img):

        return vifp(ref_img, img)
    
def save_img(img, file_name):

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    save_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Demo/', file_name)
    cv2.imwrite(save_path, img)

def post_exper(img_name):  # img_name: 612.png

    ori_img_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', img_name)
    deg_img_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_2', img_name)
    deg_S_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/S/S_2', img_name)
    deg_E_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/E/E_2', img_name)

    ori = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE) / 255
    deg = cv2.imread(deg_img_path, cv2.IMREAD_GRAYSCALE) / 255
    deg_S = cv2.imread(deg_S_path, cv2.IMREAD_GRAYSCALE) / 255
    deg_E = cv2.imread(deg_E_path, cv2.IMREAD_GRAYSCALE) / 255

    enh_img = lightening(deg_E, 1) * deg_S   ## 这个入参是方便深度学习模型确定ckt
    save_img(enh_img, 'post_gamma.png')       #### 这个地方一定要改 ####

    # 下面开始计算
    psnr = Metrics().psnr(deg, enh_img), Metrics().psnr(ori, enh_img)
    ssim = Metrics().ssim(ori, enh_img), Metrics().ssim(deg, enh_img)
    ig = Metrics().ig(deg, enh_img), Metrics().ig(ori, enh_img)
    dpi = Metrics().dpi(deg, enh_img), Metrics().dpi(ori, enh_img)
    vif = Metrics().vif(ori, enh_img), Metrics().vif(deg, enh_img)

    print(f'psnr:{psnr.item():.2f}, ssim:{ssim.item():.2f}, ig:{ig.item():.2f}, dpi:{dpi.item():.2f}, vif:{vif.item():.2f}')

def pre_exper(img_name):  # img_name: 612.png

    ori_img_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0', img_name)
    deg_img_path = os.path.join('/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_2', img_name)

    ori = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE) / 255
    deg = cv2.imread(deg_img_path, cv2.IMREAD_GRAYSCALE) / 255
    enh_img = lightening(deg)
    save_img(enh_img, 'pre_gamma.png')   #### 这个地方一定要改  ### 

    # 下面开始计算
    psnr = Metrics().psnr(deg, enh_img), Metrics().psnr(ori, enh_img)
    ssim = Metrics().ssim(ori, enh_img), Metrics().ssim(deg, enh_img)
    ig = Metrics().ig(deg, enh_img), Metrics().ig(ori, enh_img)
    dpi = Metrics().dpi(deg, enh_img), Metrics().dpi(ori, enh_img)
    vif = Metrics().vif(ori, enh_img), Metrics().vif(deg, enh_img)

    print(f'psnr:{psnr.item():.2f}, ssim:{ssim.item():.2f}, ig:{ig.item():.2f}, dpi:{dpi.item():.2f}, vif:{vif.item():.2f}')

if __name__ == '__main__':

    post_exper('612.png')
    pre_exper('612.png')











    


