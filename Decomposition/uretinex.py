import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# 定义网络结构
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)
class HalfDnCNNSE(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
         
        if self.opts.concat_L:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = self.conv1 = get_conv2d_layer(in_c=3, out_c=64, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
        self.se_layer = SELayer(channel=64)
        self.conv3 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = get_conv2d_layer(in_c=64, out_c=3, k=3, s=1, p=1)

    def forward(self, r, l):
        if self.opts.concat_L:
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
            se_inf = self.se_layer(inf)
        else:
            r_fs = self.relu1(self.conv1(r))
            se_inf = self.se_layer(r_fs)
        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n
        return r_restore
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
    
class Illumination_Alone(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=1, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv5 = get_conv2d_layer(in_c=32, out_c=1, k=1, s=1, p=0)

        self.leaky_relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_3 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
    
    def forward(self, l):
        x = l
        x1 = self.leaky_relu_1(self.conv1(x))
        x2 = self.leaky_relu_2(self.conv2(x1))
        x3 = self.leaky_relu_3(self.conv3(x2))
        x4 = self.leaky_relu_4(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        return x5   
    
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]  # 输出4通道，前3通道作为R，最后一个通道作为L
        return R, L
class P(nn.Module):
    """
        to solve min(P) = ||I-PQ||^2 + γ||P-R||^2
        this is a least square problem
        how to solve?
        P* = (gamma*R + I*Q) / (Q*Q + gamma)
    """
    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q))  
         
class Q(nn.Module):
    """
        to solve min(Q) = ||I-PQ||^2 + λ||Q-L||^2
        Q* = (lamda*L + I*P) / (P*P + lamda)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, I, P, L, lamda):
        
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]

        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]

        return (IR*PR + IG*PG + IB*PB + lamda*L) / ((PR*PR + PG*PG + PB*PB) + lamda)

def load_model():

    device = 'cuda:0'
    # load the model
    # 加载初始化模型
    model_init = Decom()
    checkpoint_init = torch.load(r'./Checkpoints/decomposition/init_low.pth', map_location='cpu')
    model_init.load_state_dict(checkpoint_init['state_dict']['model_R'])
    for param in model_init.parameters():
        param.requires_grad = False

    # 加载两个分解模型
    checkpoints = torch.load(r'./Checkpoints/decomposition/unfolding.pth', map_location=torch.device('cpu'))
    old_opts = checkpoints["opts"]
    model_R = HalfDnCNNSE(old_opts)
    model_L = Illumination_Alone(old_opts)
    model_R.load_state_dict(checkpoints['state_dict']['model_R'])
    model_L.load_state_dict(checkpoints['state_dict']['model_L'])
    for param_R in model_R.parameters():
            param_R.requires_grad = False
    for param_L in model_L.parameters():
        param_L.requires_grad = False

    model_init.to(device)
    model_R.to(device)
    model_L.to(device)

    return model_init, model_R, model_L


def decompose_one_image(input_path):

    device = 'cuda:0'
    transform = transforms.Compose([transforms.ToTensor(),])
    # 加载这张图片

    img = transform(Image.open(input_path).convert('RGB')).unsqueeze(0)
    img = img.to(device)

    # 加载计算用的组件
    model_init, model_R, model_L = load_model()
    unfold_P = P()
    unfold_Q = Q()

    # 开始迭代计算
    with torch.no_grad():     
        for t in range(3):
            if t == 0:
                P_output, Q_output = model_init(img)
            else:
                w_p = (0.1 + 0.05 * t)
                w_q = (0.5 + 0.05 * t)

                P_output = unfold_P(I=img, Q=Q_output, R=R, gamma=w_p)
                Q_output = unfold_Q(I=img, P=P_output, L=L, lamda=w_q)
                
            R = model_R(r=P_output, l=Q_output)
            L = model_L(l=Q_output)
    
    R = R.squeeze(0)
    L = L.squeeze(0)
    I = R * L

    R = transforms.ToPILImage()(R).convert('L')
    L = transforms.ToPILImage()(L).convert('L')
    I = transforms.ToPILImage()(I).convert('L')

    file_name = os.path.basename(input_path).split('.')[0]
    type_name = input_path.split('/')[-2].split('_')[-1]

    R.save(f'Results/decomposition/uretinex/{file_name}_{type_name}_S.png')
    L.save(f'Results/decomposition/uretinex/{file_name}_{type_name}_E.png')
    I.save(f'Results/decomposition/uretinex/{file_name}_{type_name}_I.png')



    # save_path_R = os.path.join(r"./Results/decomposition/uretinex/", name+"_R.jpg")
    # save_path_L = os.path.join(r'./Results/decomposition/uretinex/', name+"_L.jpg") 


    # numpy_array_R = np.array(R)
    # numpy_array_L = np.array(L)

    # img_R = np.uint8(cv2.normalize(numpy_array_R, None, 0, 255, cv2.NORM_MINMAX))
    # img_L = np.uint8(cv2.normalize(numpy_array_L, None, 0, 255, cv2.NORM_MINMAX))

    # R.save(save_path_R)
    # L.save(save_path_L)

    # cv2.imwrite('Results/decomposition/uretinex/2_R.png', img_R)
    # cv2.imwrite('Results/decomposition/uretinex/2_L.png', img_L)


def reverse_input(R_path, L_path):

    transform1 = transforms.Compose([transforms.ToTensor(),])
    transform2 = transforms.Compose([transforms.ToPILImage(),])

    img_R = transform1(Image.open(R_path))
    img_L = transform1(Image.open(L_path))

    img_out = img_R * img_L
    transform2(img_out).save(r'Images/demo/3_out.jpg')


if __name__ == '__main__':
    # checkpoints = torch.load(r'Checkpoints/decomposition/unfolding.pth', map_location=torch.device('cpu'))
    # old_opts = checkpoints["opts"]
    # print(old_opts)
    # decompose_one_image(r'./Images/I_0123/I_2/0.png')
    # reverse_input(r'Results/decomposition/uretinex/0_L.jpg',
    #               r'Results/decomposition/uretinex/0_R.jpg')

    # input_path = './Images/I_0123/I_0/0.png'
    # file_name = os.path.basename(input_path).split('.')[0]
    # type_name = input_path.split('/')[-2].split('_')[-1]
    # print(file_name, type_name)

    for i in tqdm(range(4)):
        img_path = f'./Images/I_0123/I_{i}/0.png'
        decompose_one_image(img_path)

