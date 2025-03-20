# 该实验是对比方法对比中的实验2，将增强方法称为enhan, 分解方法称为decom
# 首先验证一个猜想，对于一个红外图像增强算法，如果使用一个三通道的图像输入（原图、反射图、照射图），效果要好于只使用一个单通道的输入。将只单通道输入的模式称为a，三通道输入的模式称为b。
import os
from Denoising.CNN_based.dncnn import DnCNN
from Denoising.CNN_based.unet import U_Net
from Denoising.CNN_based.dip import skip
from Lightening.CNN_based.llnet import LLNet
from Lightening.CNN_based.deepupe import DeepUPE
from Lightening.CNN_based.llflow import RRDBNet
from Datasets.shot_by_us import SimpleDataset, RetinexDDataset, ComplexDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import random
import numpy as np
from Metrics.with_reference import psnr, ssim
import cv2
from PIL import Image


def get_enhancement_method(enhancement_method):
    if enhancement_method == 'dncnn':
        model = DnCNN()
        return model
    if enhancement_method == 'unet':
        model = U_Net()
        return model
    if enhancement_method == 'llnet':
        model = LLNet()
        return model
    if enhancement_method == 'deepupe':
        model = DeepUPE()
        return model
    if enhancement_method == 'dip':
        model = skip()
        return model
    if enhancement_method == 'llflow':
        model = RRDBNet(3, 3, 128, 3)
        return model
    
    
def get_dataset(enhancement_type, is_retinexd, other_decom):

    if not is_retinexd:
        if not other_decom:
            dataset = SimpleDataset(enhancement_type)
        if other_decom:
            dataset = ComplexDataset(enhancement_type, other_decom)

    if is_retinexd:
        dataset = RetinexDDataset(enhancement_type)
    
    return dataset

    
def train(enhancement_method, enhancement_type, device, is_retinexd, other_decom):

    model = get_enhancement_method(enhancement_method)
    model = model.to(device)
    torch.set_num_threads(2)

    dataset = get_dataset(enhancement_type, is_retinexd, other_decom)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if not is_retinexd:
        epochs = 10 if not other_decom else 12
    if is_retinexd:
        epochs = 20

    model.train()

    progress_bar = tqdm(total=epochs, desc='Training Epochs', position=0, leave=True)

    for epoch in range(epochs):
    
        total_loss = 0

        for ground_truth, focal in dataloader:
            ground_truth, focal = ground_truth.to(device), focal.to(device)

            optimizer.zero_grad()

            denoised_focal = model(focal)

            loss = criterion(ground_truth, denoised_focal)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        progress_bar.set_postfix(total_loss=total_loss / len(dataloader))
        progress_bar.update(1)  # 更新进度条

        if epoch + 1 == epochs:
            torch.save(model.state_dict(), f'Checkpoints/{enhancement_method}/{enhancement_method}_{enhancement_type}_{is_retinexd}_{other_decom}.pth')



def test_one_image(enhancement_method, enhancement_type, device, is_retinexd, other_decom):
    ### 该函数的作用仅仅是获得一个测试结果，看看训练好的模型是不是可以正常工作
    ### 不能获取指标

    ### 第一步，我们计算出权重文件的路径
    ckt_name = f'{enhancement_method}_{enhancement_type}_{is_retinexd}_{other_decom}.pth'
    ckt_path = os.path.join('Checkpoints', enhancement_method, ckt_name)

    ### 第二步，加载出必要的组件
    model = get_enhancement_method(enhancement_method)
    model.load_state_dict(torch.load(ckt_path))
    model = model.to(device)
    torch.set_num_threads(2)

    dataset = get_dataset(enhancement_type, is_retinexd, other_decom)

    # random.seed(42)
    # random_int = random.randint(1, 1024)
    random_int = 229

    groundtruth, focal = dataset[random_int]

    focal = focal.unsqueeze(0).to(device)

    ### 第三步，增强退化图像
    model.eval()
    with torch.no_grad():
        enhanced_focal = model(focal)
        enhanced_focal = enhanced_focal.squeeze().cpu()

    ### 第3.5步，保存增强结果
    numpy_image = (enhanced_focal * 255).byte().numpy()
    numpy_image = np.transpose(numpy_image, (1, 2, 0))
    numpy_image = np.mean(numpy_image, axis=2)
    cv2.imwrite(f'Results/experiment2.2/{enhancement_method}/{enhancement_method}_{enhancement_type}_{is_retinexd}_{other_decom}.png', numpy_image)

    ### 第四步，准备保存实验结果
    groundtruth = torch.mean(groundtruth, dim=0)
    enhanced_focal = torch.mean(enhanced_focal, dim=0)
    groundtruth, enhanced_focal = groundtruth.numpy(), enhanced_focal.numpy()

    ### 第五步，保存实验结果
    focal = torch.mean(focal.squeeze(), dim=0).cpu().numpy()

    if not os.path.exists(f'Results/experiment2.2/{enhancement_method}/groundtruth.png'):
        cv2.imwrite(f'Results/experiment2.2/{enhancement_method}/groundtruth.png', (groundtruth * 255).astype(np.uint8))

    if not os.path.exists(f'Results/experiment2.2/{enhancement_method}/input.png'):
        cv2.imwrite(f'Results/experiment2.2/{enhancement_method}/input.png', (focal * 255).astype(np.uint8))


def measure(enhancement_method, enhancement_type, device, is_retinexd, other_decom):
    ### 该函数的作用是计算客观指标

    ### 第一步，我们计算出权重文件的路径
    ckt_name = f'{enhancement_method}_{enhancement_type}_{is_retinexd}_{other_decom}.pth'
    ckt_path = os.path.join('Checkpoints', enhancement_method, ckt_name)

    ### 第二步，加载出必要的组件
    model = get_enhancement_method(enhancement_method)
    model.load_state_dict(torch.load(ckt_path))
    model = model.to(device)
    torch.set_num_threads(2)

    dataset = get_dataset(enhancement_type, is_retinexd, other_decom)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    groundtruth, focal = next(iter(dataloader))

    focal = focal.to(device)

    ### 第三步，增强退化图像
    model.eval()
    with torch.no_grad():
        enhanced_focal = model(focal)

    ### 第四步，计算指标
    psnr_vals = np.zeros(50)
    ssim_vals = np.zeros(50)
    for i in range(50):
        y, x = groundtruth[i], enhanced_focal[i].cpu()
        y = np.mean(y.numpy(), axis=0)
        x = np.mean(x.numpy(), axis=0)
        psnr_vals[i] = psnr(y, x)
        ssim_vals[i] = ssim(y, x)
    print(f'psnr:{np.mean(psnr_vals)}, ssim:{np.mean(ssim_vals)}')




if __name__ == '__main__':
    # 调整参数区
    enhancement_method = 'dip'
    enhancement_type = 'lightening'
    device = 'cuda:0'
    is_retinexd = True
    other_decom = None   

    # 执行代码

    # train(enhancement_method, enhancement_type, device, is_retinexd, other_decom)

    test_one_image(enhancement_method, enhancement_type, device, is_retinexd, other_decom)

    measure(enhancement_method, enhancement_type, device, is_retinexd, other_decom)






