import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import argparse
import random
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn

pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V,H) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * 1.3
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
 
# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x


class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
parser = argparse.ArgumentParser()
parser.add_argument('--need_resize', type=bool, default=True)  # 是否需要resize, 如果需要resize，一律改到256大小
parser.add_argument('--img_names', type=list[str], default=random.sample(os.listdir(r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I/I_0'), 2785)) # 获取打乱顺序的文件列表
parser.add_argument('--I_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/I')
parser.add_argument('--S_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/S')
parser.add_argument('--E_folder_path', type=str, default=r'/mnt/jixie8t/zd_new/Code/RetinexD/Images/E')
parser.add_argument('--py_filename', type=str, default=os.path.splitext(os.path.basename(os.path.abspath(sys.argv[0])))[0])
parser.add_argument('--img_type', type=str, default="RGB")
args = parser.parse_args()


class VariableDataset(Dataset): 
    def __init__(self, decompose_mode, denoising_mode):
        self.img_names = args.img_names[0:2735]  # 取前2735张图片作为训练集
        self.decompose_mode = decompose_mode
        self.denoising_mode = denoising_mode
        if args.need_resize:
            self.transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()
    def __len__(self):
        return 2735
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        degraded_I_path = os.path.join(args.I_folder_path, 'I_'+self.denoising_mode[-1], img_name)
        groundtruth_I_path = os.path.join(args.I_folder_path, 'I_'+self.denoising_mode[0], img_name)
        if self.decompose_mode == None:
            return self.transform(Image.open(degraded_I_path).convert(args.img_type)), self.transform(Image.open(groundtruth_I_path).convert(args.img_type))   # I1, I0
        if self.decompose_mode == 'retinexd':
            S_img_path = os.path.join(args.S_folder_path, 'S_'+self.denoising_mode[-1], img_name)
            E_img_path = os.path.join(args.E_folder_path, 'E_'+self.denoising_mode[-1], img_name)
            return self.transform(Image.open(degraded_I_path).convert(args.img_type)), self.transform(Image.open(E_img_path).convert(args.img_type)), self.transform(Image.open(S_img_path).convert(args.img_type)), self.transform(Image.open(groundtruth_I_path).convert(args.img_type))   # I1, E1, S1, I0

def variabletrain(device, decompose_mode, denoising_mode): 
    model = CIDNet()      # 为了尽可能快的训练，使用单卡训练
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60 if decompose_mode == 'retinexd' else 30
    dataset = VariableDataset(decompose_mode=decompose_mode, denoising_mode=denoising_mode)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        if decompose_mode is None:
            for degraded_I, groundtruth_I in progress_bar:
                degraded_I, groundtruth_I = degraded_I.to(device), groundtruth_I.to(device)
                optimizer.zero_grad()
                enhanced_I = model(degraded_I)
                loss = criterion(enhanced_I, groundtruth_I)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 5) == 0:
                torch.save(model.state_dict(), f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/{args.py_filename}/{args.py_filename}_{denoising_mode}.pth')
        else:
            for degraded_I, E, S, groundtruth_I in progress_bar:
                degraded_I, E, S, groundtruth_I = degraded_I.to(device), E.to(device), S.to(device), groundtruth_I.to(device)
                optimizer.zero_grad()
                enhanced_I, enhanced_E = model(degraded_I), model(E)
                loss = criterion(S*enhanced_E, groundtruth_I) + criterion(enhanced_I, S*enhanced_E)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            if epoch+1 == epochs or (epoch % 5) == 0:
                save_name = f'{args.py_filename}_{denoising_mode}_with_' + decompose_mode + '.pth'
                torch.save(model.state_dict(), os.path.join(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/{args.py_filename}/', save_name))

def cidnet(img, decompose_mode=None):
    model = CIDNet()
    if decompose_mode == None:
        model.load_state_dict(torch.load(f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/cidnet/cidnet_0to2.pth'))
    else:
        ckt_path = f'/mnt/jixie8t/zd_new/Code/RetinexD/Checkpoints/lightening/cidnet/cidnet_0to2_with_retinexd.pth'
        model.load_state_dict(torch.load(ckt_path))
    model.eval()
    input_array = img
    tensor = torch.tensor(input_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0) # tensor: [1, 1, 360, 500]
    repeated_tensor = tensor.repeat(1, 3, 1, 1)
    resized_tensor = torch.nn.functional.interpolate(repeated_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    with torch.no_grad():
        tensor = model(resized_tensor)
    deresized_tensor = torch.nn.functional.interpolate(tensor, size=(360, 500), mode='bilinear', align_corners=False)
    derepeated_tensor = deresized_tensor.squeeze()[0]
    
    return derepeated_tensor.cpu().numpy()
    

if __name__ == '__main__':
    # model = CIDNet().cuda()
    # x = torch.randn(1, 3, 256, 256).cuda()
    # y = model(x).cpu()
    # print(y.shape)
    variabletrain('cuda:1', decompose_mode='retinexd', denoising_mode='0to2')

    
    