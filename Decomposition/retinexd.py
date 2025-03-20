import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def decompose(img_path):
    # 该函数接收一张图片的路径，输出两个张量，即SSR分解结果的反射域图与照射景图
    # 图片路径 -> [1, H, W], [1, H, W]
    img = cv2.imread(img_path, 0)  
    sigma = 25
    img = np.where(img==0, 1e-10, img)

    illumination = cv2.GaussianBlur(img, (0, 0), sigma)
    reflection = np.power(10, np.log10(img) - np.log10(illumination))

    illumination = np.uint8(cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX))
    reflection = np.uint8(cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX))

    illumination = torch.from_numpy(illumination).float() / 255.0
    reflection = torch.from_numpy(reflection) / 255.0

    return reflection.unsqueeze(0), illumination.unsqueeze(0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class HalfDnCNNSE(nn.Module):
    # 反射图的分解网络
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 32, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)

        self.se_layer = SELayer(channel=64)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, r, l):

        r_fs = self.relu1(self.conv1(r))
        l_fs = self.relu2(self.conv2(l))

        inf = torch.cat([r_fs, l_fs], dim=1)
        se_inf = self.se_layer(inf)

        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n

        return r_restore
    
class P_or_Q(nn.Module):
    """
        to solve min(P) = ||I-PQ||^2 + γ||P-R||^2
        this is a least square problem
        how to solve?
        P* = (gamma*R + I*Q) / (Q*Q + gamma)
    """
    # 此为求解工程量P、Q的模块，当输入Q的位置变成P，R的位置变成E，gamma的位置变成lambda,
    # 变成Q的求解模块
    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):

        return ((I * Q + gamma * R) / (gamma + Q * Q))  

class BaseData(Dataset):
    # 一个最基础的数据集，实例化时给定一个文件夹的目录, 返回该文件夹下图片的SSR分解量r,l
    def __init__(self, folder_path):
        super().__init__()
        self.image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
        
        self.transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):

        return len(self.image_files)
    
    def __getitem__(self, index):

        image_path = self.image_files[index]
        tensor_r, tensor_l = decompose(image_path) 
        i = self.transform(Image.open(image_path).convert('L'))


        return tensor_r, tensor_l, i

def train_for_one():

    # 此训练的目的是使每一次迭代的结果都能够与工程量接近
    model_R = HalfDnCNNSE()
    model_L = HalfDnCNNSE() # 尝试将两者换成相同的网络

    model_unfold = P_or_Q()

    # model_R.load_state_dict(torch.load('./Checkpoints/decomposition/R.pth'))
    # model_L.load_state_dict(torch.load('./Checkpoints/decomposition/L.pth'))

    # 训练发现model_R的训练效果较好，停止R的梯度更新，单独训练L
    # for param in model_R.parameters():
    #     param.requires_grad = False

    if torch.cuda.is_available():

        device = torch.device('cuda:0')
        model_R = model_R.to(device)
        model_L = model_L.to(device)

        model_R = nn.DataParallel(model_R, device_ids=[0, 1, 2, 3])
        model_L = nn.DataParallel(model_L, device_ids=[0, 1, 2, 3])

    criterion = nn.MSELoss()
    optimizer_R = torch.optim.Adam(model_R.parameters(), lr=1e-4)
    optimizer_L = torch.optim.Adam(model_L.parameters(), lr=1e-4)

    dataset = BaseData(r'./Images/I_0123/I_0/')
    batchs_size = 36
    dataloader = DataLoader(dataset, batch_size=batchs_size, shuffle=True)

    epochs = 20
    for epoch in range(epochs):

        model_R.train()
        model_L.train()
        total_loss_R = 0.0
        total_loss_L = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for r, l, i in progress_bar:

            r, l, i = r.to(device), l.to(device), i.to(device)

            optimizer_R.zero_grad()
            optimizer_L.zero_grad()

            # 记录第一次的损失函数，此次没有迭代过程
            R = model_R(r, i)
            L = model_L(l, i)

            loss_R = criterion(R, r.detach())
            loss_L = criterion(L, l.detach())


            # 记录第二次的损失函数，发生第一次迭代

            r = model_unfold(I=i, Q=l, R=R.detach(), gamma=0.15)
            l = model_unfold(I=i, Q=r, R=L.detach(), gamma=0.55)

            R = model_R(r, i)
            L = model_L(l, i)

            loss_R = loss_R + criterion(R, r.detach())
            loss_L = loss_L + criterion(L, l.detach())


            # 记录第三次的损失函数，发生第三次迭代

            r = model_unfold(I=i, Q=l, R=R.detach(), gamma=0.2)
            l = model_unfold(I=i, Q=r, R=L.detach(), gamma=0.6)

            R = model_R(r, i)
            L = model_L(l, i)

            loss_R = loss_R + criterion(R, r.detach())
            loss_L = loss_L + criterion(L, l.detach())

            total_loss_R += loss_R.item()
            total_loss_L += loss_L.item()

            loss_R.backward()
            loss_L.backward()
            optimizer_R.step()
            optimizer_L.step()

            progress_bar.set_postfix(loss_R=loss_R.item(), loss_L=loss_L.item())

        avg_loss_R, avg_loss_L = total_loss_R / (len(dataloader) * batchs_size * 3), total_loss_L / (len(dataloader) * batchs_size * 3)

        with open(f"./Log/decompose/decompose_first_phase.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1}/{epochs}, Average Loss R: {avg_loss_R:.10f}, Average Loss L:{avg_loss_L:.10f} \n")

        if (epoch+1) % 1 == 0 or epoch+1 == epochs:
            torch.save(model_R.module.state_dict(), f'./Checkpoints/decomposition/R_new.pth')
            torch.save(model_L.module.state_dict(), f'./Checkpoints/decomposition/L_new.pth')

def test_for_one_phase():

    # 此函数的作用是验证网络的分解效果
    model_R = HalfDnCNNSE()
    model_L = HalfDnCNNSE()
    model_unfold = P_or_Q()

    model_R.load_state_dict(torch.load('./Checkpoints/decomposition/R_new.pth'))
    model_L.load_state_dict(torch.load('./Checkpoints/decomposition/L_new.pth'))

    for param in model_R.parameters():
        param.requires_grad = False
    for param in model_L.parameters():
        param.requires_grad = False

    device = torch.device('cuda:0')
    model_R = model_R.to(device)
    model_L = model_L.to(device)

    # 在这里选取需要分解的图片
    img_path = './Images/I_0123/I_0/0.png'
    r, l = decompose(img_path)
    i = transforms.Compose([transforms.ToTensor(),])(Image.open(img_path).convert('L'))

    model_R.eval()
    model_L.eval()

    r, l, i = r.to(device).unsqueeze(0), l.to(device).unsqueeze(0), i.to(device).unsqueeze(0)

    with torch.no_grad():

        R = model_R(r, i)
        L = model_L(l, i)

        r = model_unfold(I=i, Q=l, R=R.detach(), gamma=0.15)
        l = model_unfold(I=i, Q=r, R=L.detach(), gamma=0.55)

        R = model_R(r, i)
        L = model_L(l, i)

        r = model_unfold(I=i, Q=l, R=R.detach(), gamma=0.2)
        l = model_unfold(I=i, Q=r, R=L.detach(), gamma=0.6)

        R = model_R(r, i)
        L = model_L(l, i)

    to_pil = transforms.ToPILImage(mode='L')
    R_image = to_pil(R.squeeze())
    L_image = to_pil(L.squeeze())
    I_image = to_pil(R.squeeze() * L.squeeze())
    R_image.save('./Images/demo/S0_new.png')
    L_image.save('./Images/demo/E0_new.png')
    I_image.save('./Images/demo/I0_new.png')
    





if __name__ == "__main__":

    # train_for_one()
    # dataset = BaseData(r'./Images/I_0123/I_0/')
    # x, y, z = dataset[0]
    # print(x.shape, y.shape, z.shape)

    test_for_one_phase()

    