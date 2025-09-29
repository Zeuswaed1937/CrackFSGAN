import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

# SLE模块的实现，Squeeze-and-Excitation Block的一类变体，可以接受2个特征图
class SEBlock(nn.Module):
    def __init__(self, 
                 ch_in, #输入特征图的通道数
                 ch_out #输出特征图的通道数
                 ):
        super().__init__()
        # 包括自适应平均池化、卷积、Swish激活和另一个卷积以及Sigmoid激活
        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        # 首先对feat_small应用一系列操作，
        # 得到一个形状为 (batch_size, ch_out, 1, 1) 的张量
        # 这个张量中的每个元素代表相应通道的重要性权重
        # 然后将这些权重应用于较大的特征图 feat_big，通过逐元素相乘的方式
        # 即每个通道的特征都被相应的权重所缩放
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    # 以顺序的方式封装了一个或多个模块（如层或操作），使得数据可以按照定义的顺序通过这些模块进行前向传播
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block


# 定义原始的生成器Class
class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])
                                
        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False) 
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False) 
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  
        
    def forward(self, input):
        
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)

        # 调用SLE模块进行处理
        feat_64  = self.se_64( feat_4, self.feat_64(feat_32) )

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )

        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]
        
        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )

        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]

# 定义简单下采样块
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), 
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)

# 定义复合式下采样块
class DownBlockComp(nn.Module):
    def __init__(self, 
                 in_planes, # 输入的特征图通道数
                 out_planes # 输出的特征图通道数
                 ):
        # 初始化函数，定义该Class的层和参数
        super(DownBlockComp, self).__init__()
# 标准的卷积下采样路径
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            # 使用 4x4 的卷积核，步长为 2，填充为 1，进行下采样操作
            # 该层将输入特征图从in_planes个通道转换为out_planes个通道，并减半空间分辨率
            batchNorm2d(out_planes), 
            # 对输出特征图进行批归一化，帮助加速训练并稳定模型
            nn.LeakyReLU(0.2, inplace=True),
            # 应用 LeakyReLU 激活函数，负斜率为 0.2，采用原地计算以节省内存
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            # 使用 3x3 卷积核，步长为 1，填充为 1，保持空间分辨率不变，但进一步变换特征
            batchNorm2d(out_planes), 
            nn.LeakyReLU(0.2)
            )
# 直接下采样后再进行卷积调整通道数的路径
        self.direct = nn.Sequential(
            # 使用平均池化层将输入特征图的空间分辨率减半
            nn.AvgPool2d(2, 2), 
            # 使用1x1卷积核调整通道数，从in_planes转换为out_planes，不改变空间分辨率
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            # 批归一化层
            batchNorm2d(out_planes), 
            # 应用 LeakyReLU 激活函数，负斜率为 0.2，采用原地计算以节省内存
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        # 定义数据通过网络的路径，取所定义的两种路径的平均值
        return (self.main(feat) + self.direct(feat)) / 2

# 定义原始鉴别器Class
class Discriminator(nn.Module):
    # 在PyTorch的nn.Module类中，__call__方法被重载以调用forward()方法，并且还添加了一些额外的功能
    def __init__(self, 
                 ndf=64, 
                 nc=3, #默认为3，事彩色图像
                 im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf # 特征图的数量，默认64不改变
        # nc是图的通道的数量
        self.im_size = im_size # 图像尺寸

        # 根据图像尺寸调整特征图数量的规则的系数
        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {} # 特征图的通道的数量，是一个字典，key是图像的尺寸，value是通道数量
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf) # 对于256分辨率，则为0.5*64

        # 定义大型尺寸图像的下采样策略，根据输入图像的尺寸修改
        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False), # 步长为2
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False), # 步长为2
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False), # 步长为2
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False), # 步长为1
                                    nn.LeakyReLU(0.2, inplace=True))

        # 下采样Module，通道数变大了，对于的size变小
        self.down_4  = DownBlockComp(nfc[512], nfc[256]) # I型下采样，16 32 
        self.down_8  = DownBlockComp(nfc[256], nfc[128]) # II型下采样，32 64
        self.down_16 = DownBlockComp(nfc[128], nfc[64])  # III型下采样，64 128
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])  # IV型下采样，128 256
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])  # V型下采样，256 512

        # Receptive Field，用于最终输出原始图像的预测分数
        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))



        self.se_2_16 = SEBlock(nfc[512], nfc[64]) # I型SLE模块，16 128
        self.se_4_32 = SEBlock(nfc[256], nfc[32]) # II型SLE模块，32 256
        self.se_8_64 = SEBlock(nfc[128], nfc[16]) # III型SLE模块，64 512
        
        # 小型尺寸图像的下采样策略
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )


        # Receptive Field，用于最终输出缩小后图像的预测分数
        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        # 3种解码器，从不同尺寸的特征中重建图像
        # 原始图像用解码器
        self.decoder_big = SimpleDecoder(nfc[16], nc) # 512 3
        # 部分图像用解码器
        self.decoder_part = SimpleDecoder(nfc[32], nc) # 256 3
        # 小尺寸图像用解码器
        self.decoder_small = SimpleDecoder(nfc[32], nc) # 256 3
        
    # 前向传播函数是被直接调用的，直接定义鉴别器网络执行的流程
    # imgs是单张的图像
    def forward(self, imgs, label, part=None):
        # 插值法预处理传入图像，锁定为列表格式，为原始图和缩小后的图像
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        # 处理原始尺寸图像
        feat_2 = self.down_from_big(imgs[0]) # 直接处理增强处理后的原始图像了       
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8) # 系列下采样
        feat_16 = self.se_2_16(feat_2, feat_16) # SLE处理

        feat_32 = self.down_32(feat_16) # 系列下采样
        feat_32 = self.se_4_32(feat_4, feat_32) # SLE处理
        
        # 最终特征图
        feat_last = self.down_64(feat_32) # 系列下采样
        feat_last = self.se_8_64(feat_8, feat_last) # SLE处理


        # 原始尺寸的判别分数
        rf_0 = self.rf_big(feat_last).view(-1)

        # 处理缩小后的图像
        feat_small = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        # 缩小的尺寸判别分数
        rf_1 = self.rf_small(feat_small).view(-1)

        if label=='real': # label是一个可传入的参数，判断是否为真实图像   
            rec_img_big = self.decoder_big(feat_last) # 返回原始图像重建值
            # size --> (batch size, 3, 256, 256)

            rec_img_small = self.decoder_small(feat_small) # 返回缩小后图像的重建值
            # size --> (batch size, 3, 128, 128)


            # part是随机数，0-4
            assert part is not None
            rec_img_part = None
            # size --> (batch size, 3, 32, 32)
            # 根据随机数不同选用不同位置，返回重建的part图像，即随机重建一部分
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]
            # 是真实图像，返回值则增添了重构的图像

        return torch.cat([rf_0, rf_1]) # size --> (batch size*2,)

# 定义简单解码器给鉴别器使用
class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        # 特征图默认通道数，调用时，传入的特征图默认通道数根据解码器类型而改变
        # 图像通道数
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {} # 特征图的通道的数量，是一个字典，key是图像的尺寸，value是通道数量
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        # 定义了一个上采样模块
        def upBlock(in_planes, # 输入特征图的通道数
                    out_planes # 输出特征图的通道数
                    ):
            # 
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                                    # in_channels=in_planes, 输入通道数，out_channels=out_planes*2, 和输出通道数
                                    # kernel_size=3, 使用3x3的卷积核，
                                    # stride=1, 步长为1，
                                    # padding=1, 填充1个像素以保持特征图的空间尺寸不变 
                batchNorm2d(out_planes*2), 
                GLU() # 将输入分成两半，一半作为门控信号，另一半经过线性变换后与门控信号相乘
                )
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) , # 根据给定特征图的通道数进行上采样
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                   
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


# 定义随机裁剪，这个函数并没有投入实际使用
from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]


# 定义纹理鉴别器Class，这个Class并没有投入实际使用
class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf