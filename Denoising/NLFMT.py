import cv2
# import pywt
import math
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm 
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
import torch
import pyiqa


def nlmeans_filt2D(xn_path, sigmas=5, ksize=7, ssize=21):

   xn = cv2.imread(xn_path, cv2.IMREAD_GRAYSCALE)

   # 进行离散小波变换, 目的是估计出噪声的标准差
   coeffes = pywt.dwt2(xn, 'db8')
   _, (_, _, cd) = coeffes
   tt1 = cd.flatten()   # cd为离散小波变换的对角细节系数，转换为一维数组

   median_hh2 = np.median(np.abs(tt1)) 
   std_dev2 = (median_hh2 / 0.6745)

   noise_std = std_dev2

   half_ksize = math.floor(ksize/2)       # 3
   half_ssize = math.floor(ssize/2)       # 10

   M, N = xn.shape                        # 360, 500

   # 定义xm
   xm = np.zeros((M+ssize-1, N+ssize-1))
   xm[half_ssize:M+half_ssize, half_ssize:N+half_ssize] = xn  # 中心区域

   xm[0:half_ssize, :] = xm[ssize-1:half_ssize:-1, :]
   xm[M+half_ssize:M+ssize-1, :] = xm[M+half_ssize-1:M-1:-1, :]  # 边缘行

   xm[:, 0:half_ssize] = xm[:, ssize-1:half_ssize:-1]
   xm[:, N+half_ssize:N+ssize-1] = xm[:, N+half_ssize-2:N-2:-1] # 边缘列

   # 定义高斯滑动的窗口
   gauss_win = gaussian_filter(np.zeros((ksize, ksize)), sigma=sigmas)

   # NL-means Filter Implementation.
   filt_h = 0.55 * noise_std
   M, N = xm.shape
   im_rec = np.zeros((M - 2 * half_ssize, N - 2 * half_ssize))
   for ii in range(half_ssize + 1, M - half_ssize):
      for jj in range(half_ssize + 1, N - half_ssize):
         xtemp = xm[ii - half_ksize:ii + half_ksize + 1, jj - half_ksize:jj + half_ksize + 1]
         search_win = xm[ii - half_ssize:ii + half_ssize + 1, jj - half_ssize:jj + half_ssize + 1]
         weight = np.zeros((ssize - ksize + 1, ssize - ksize + 1))

         # 循环计算每个像素位置的权重
         for kr in range(ssize - ksize + 1):
            for kc in range(ssize - ksize + 1):
                euclid_dist = (xtemp - search_win[kr:kr + ksize, kc:kc + ksize]) ** 2
                wt_dist = gauss_win * euclid_dist
                sq_dist = np.sum(np.sum(wt_dist)) / (ksize**2)
                weight[kr, kc] = np.exp(-max(sq_dist - (2 * noise_std**2), 0) / filt_h**2)

         sum_wt = np.sum(np.sum(weight))
         weightn = weight / sum_wt
         sum_pix = np.sum(np.sum(search_win[half_ksize:ssize - half_ksize, half_ksize:ssize - half_ksize] * weightn))
         im_rec[ii - half_ssize - 1, jj - half_ssize - 1] = sum_pix

   return im_rec

def nlmeans_filt2D_mnt(img_path):

   xn = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # 读取原始图片
   img_rec = np.load('./Results/denoise/NLFMT/0_I.npy')  # 读取初步去噪图片

   yd = xn.astype(np.float32) - img_rec

   # 执行二维小波分解
   pywt.MODES.default = 'periodization'
   coeffs = pywt.wavedec2(yd, wavelet='db8', level=3)
   # 处理分解得到的系数
   CW = [None] * 10
   C = []
   S = []
   for coeff in coeffs:
      if isinstance(coeff, tuple):
        # 对于细节系数
         for subband in coeff:
            C.extend(np.ravel(subband))
            S.append(subband.shape)
      else:
        # 对于逼近系数
         C.extend(np.ravel(coeff))
         S.insert(0, coeff.shape)  # 逼近系数尺寸在S的开始位置
   C = np.array(C)

   # 提取和重塑小波系数
   k = 10
   st_pt = 0
   for i, size in enumerate(S):
      slen = size[0] * size[1]
      if i == 0:
         # 逼近系数
         CW[k-1] = C[st_pt:st_pt+slen].reshape(size)
      else:
         # 细节系数
         CW[k-3] = C[st_pt:st_pt+slen].reshape(size)          # 水平
         CW[k-2] = C[st_pt+slen:st_pt+2*slen].reshape(size)   # 垂直
         CW[k-1] = C[st_pt+2*slen:st_pt+3*slen].reshape(size) # 对角线
         k -= 3
      st_pt += slen * (3 if i != 0 else 1)

   tt2 = CW[0].ravel()

   # 计算这些系数的绝对值的中值
   median_hh2 = np.median(np.abs(tt2))
   # 用中值估算标准差
   std_dev2 = median_hh2 / 0.6745
   # 计算噪声方差（方差是标准差的平方）
   cw_noise_var = std_dev2 ** 2

   yw = [None] * 10  # 初始化结果列表

   def bayesthf(y, noise_var):
      p, q = y.shape
      meany = np.sum(y) / (p * q)
      vary = np.sum((y - meany) ** 2) / (p * q)

      tt = vary - noise_var
      if tt < 0:
         tt = 0
      sigmax = np.sqrt(tt)
      thr = noise_var / sigmax
      if thr == np.inf:
         thr = np.max(np.abs(y))

      return thr

   for i in range(9):
      # 计算贝叶斯阈值
      thr = bayesthf(CW[i], cw_noise_var)

      # 应用阈值
      # 使用pywt.threshold函数，mode取决于sorh的值
      yw[i] = pywt.threshold(CW[i], thr, mode='s')

   # 将最后一个子带（逼近系数）直接复制
   yw[9] = CW[9]

   # 小波重组
   k = 10
   xrtemp = yw[k-1].ravel()
   k -= 1

   for i in range(1, len(S)-1):
      xrtemp = np.concatenate([xrtemp, 
                              yw[k-1].ravel(), 
                              yw[k].ravel(), 
                              yw[k-2].ravel()])
      k -= 3

   coeffs = pywt.array_to_coeffs(xrtemp, S, output_format='wavedec2')
   ydr = pywt.waverec2(coeffs, wavelet='db8')

   nl_mnt = img_rec + ydr
   
   return nl_mnt

def caculate_mse(path_x, path_y):

   x = cv2.imread(path_x, cv2.IMREAD_GRAYSCALE)
   y = cv2.imread(path_y, cv2.IMREAD_GRAYSCALE)

   err = np.sum((x.astype("float") - y.astype("float")) ** 2)
   err /= float(x.shape[0] * x.shape[1])  # 平均到每个像素上的平方误差

   return err

def caculate_psnr(path_x, path_y):
   
   mse_value = caculate_mse(path_x, path_y)

   if mse_value == 0:
      return float('inf')
   
   max_pixel = 255.0
   psnr_single_image = 20 * np.log10(max_pixel / np.sqrt(mse_value))

   return psnr_single_image

def caculate_psnr_for_folder(folder_path_x, folder_path_y):
    
   list = []
   for i in tqdm(range(1, 10)):

      image_path_x = os.path.join(folder_path_x, f'{i}_I_norm.png')
      image_path_y = os.path.join(folder_path_y, f'{i}.png')

      psnr = caculate_psnr(image_path_x, image_path_y)        
      list.append(psnr)

   numpy_list= np.array(list)
   avg_psnr = np.mean(numpy_list)

   return avg_psnr

def caculate_ssim_for_folder(folder_path_x, folder_path_y):
    
   list = []
   for i in tqdm(range(1, 10)):

      image_path_x = os.path.join(folder_path_x, f'{i}_I_norm.png')
      image_path_y = os.path.join(folder_path_y, f'{i}.png')

      img_x = cv2.imread(image_path_x, cv2.IMREAD_GRAYSCALE) 
      img_y = cv2.imread(image_path_y, cv2.IMREAD_GRAYSCALE) 

      ssim_value = compare_ssim(img_x, img_y, full=False)
      list.append(ssim_value)
    
   numpy_list= np.array(list)
   avg_ssim = np.mean(numpy_list)

   return avg_ssim

def caculate_snr_for_folder(folder_path_x, folder_path_y):
    
   list = []
   for i in tqdm(range(1, 10)):

      image_path_x = os.path.join(folder_path_x, f'{i}_I_norm.png')
      image_path_y = os.path.join(folder_path_y, f'{i}.png')

      img_x = cv2.imread(image_path_x, cv2.IMREAD_GRAYSCALE) 
      img_y = cv2.imread(image_path_y, cv2.IMREAD_GRAYSCALE) 

      signal_power = np.mean(img_y ** 2)
      noise_power = np.mean((img_x - img_y) ** 2)

      snr = 10 * np.log10(signal_power / noise_power)
      list.append(snr)
    
   numpy_list= np.array(list)
   avg_snr = np.mean(numpy_list)

   return avg_snr

def caculate_niqe_for_folder(folder_path):
    # 计算文件夹下的平均niqe，为节省计算时间，只选取1000张
    # 调用pyiqa库

   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   iqa_metric = pyiqa.create_metric('niqe', device=device)

   list = []
    
   for i in tqdm(range(1, 10)):
        
      img_path = os.path.join(folder_path, f'{i}_I_norm.png')
      niqe = iqa_metric(img_path)
      
      number = niqe.cpu().item()
      list.append(number)

   numpy_list = np.array(list)
   avg_niqe = np.mean(numpy_list)

   return avg_niqe

   

if __name__ == '__main__':

   # im_rec = nlmeans_filt2D('./Images/S_0123/S_1/0.png')
   # np.save("./Results/denoise/NLFMT/0_S.npy", im_rec)
   # img = np.load("./Results/denoise/NLFMT/0_S.npy")
   # print(img)
   # img_norm = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
   # cv2.imwrite("./Results/denoise/NLFMT/0_S.png", img)

   # img = nlmeans_filt2D_mnt('./Images/I_0123/I_1/0.png')
   # np.save("./Results/denoise/NLFMT/0_I_mnt.npy", img)

   # 程序耗时太长，仅跑10张测试图片
   # for i in range(1,10):

   #    # 先对I去噪
   #    im_rec = nlmeans_filt2D(f'./Images/I_0123/I_1/{i}.png')
   #    np.save(f"./Results/denoise/NLFMT/{i}_I.npy", im_rec)
   #    img = np.load(f"./Results/denoise/NLFMT/{i}_I.npy")
   #    img_norm = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
   #    cv2.imwrite(f"./Results/denoise/NLFMT/{i}_I.png", img)
   #    cv2.imwrite(f"./Results/denoise/NLFMT/{i}_I_norm.png", img_norm)

   #    # 再对S去噪
   #    im_rec = nlmeans_filt2D(f'./Images/S_0123/S_1/{i}.png')
   #    np.save(f"./Results/denoise/NLFMT/{i}_S.npy", im_rec)
   #    img = np.load(f"./Results/denoise/NLFMT/{i}_S.npy")
   #    img_norm = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
   #    cv2.imwrite(f"./Results/denoise/NLFMT/{i}_S.png", img)
   #    cv2.imwrite(f"./Results/denoise/NLFMT/{i}_S_norm.png", img_norm)

   avg_psnr = caculate_psnr_for_folder('./Results/denoise/NLFMT', './Images/I_0123/I_0')
   print(avg_psnr)

   # avg_ssim = caculate_ssim_for_folder('./Results/denoise/NLFMT', './Images/I_0123/I_0')
   # print(avg_ssim)  

   # avg_snr = caculate_snr_for_folder('./Results/denoise/NLFMT', './Images/I_0123/I_0')
   # print(avg_snr)  

   # niqe = caculate_niqe_for_folder('./Results/denoise/NLFMT')
   # print(niqe)
















