import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy import linalg
import numpy as np
from skimage import img_as_ubyte
from torchvision import transforms

import LPIPS_calculation
percept = LPIPS_calculation.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True) # 加载VGG作为特征提取器

def calcute_lpips(real_image, fake_image, batch_size):
    lpips_values = []
    for i in range(batch_size):
        lpips_value_i = percept(real_image[i].unsqueeze(0), fake_image[i].unsqueeze(0)).mean().item()
        lpips_values.append(lpips_value_i)
    avg_lpips_value = sum(lpips_values) / len(lpips_values)  # 取平均, a float
    return avg_lpips_value

# 计算FID的函数
def calculate_fid(real_images, fake_images, model, device):
    # 确保输入是[8, 3, 1024, 1024]的格式
    # 图像大小调整到Inception V3所需的299x299
    resize = transforms.Resize((299, 299))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # InceptionV3的标准归一化
    # 对输入图像进行大小调整并归一化
    real_images1 = torch.stack([normalize(resize(image)) for image in real_images])
    fake_images1 = torch.stack([normalize(resize(image)) for image in fake_images])
    real_images2 = real_images1.to(device) # 将图像传入模型进行特征提取
    fake_images2 = fake_images1.to(device)
    with torch.no_grad(): # 模型前向推理
        real_features = model(real_images2)
        fake_features = model(fake_images2)
    real_mu = torch.mean(real_features, dim=0) # 计算真实图像和假图像特征的均值和协方差
    real_sigma = torch.cov(real_features.T)
    fake_mu = torch.mean(fake_features, dim=0)
    fake_sigma = torch.cov(fake_features.T)
    # 将矩阵移至CPU并转换为NumPy数组
    real_sigma_cpu = real_sigma.cpu().numpy()
    fake_sigma_cpu = fake_sigma.cpu().numpy()

    diff_mu = real_mu - fake_mu # 计算FID
    diff_mu_cpu = diff_mu.cpu().numpy()  # 转换为NumPy数组

    cov_sqrt, _ = linalg.sqrtm(fake_sigma_cpu @ real_sigma_cpu, disp=False)
    if not np.isfinite(cov_sqrt).all(): # 检查奇异矩阵并处理
        offset = np.eye(real_sigma_cpu.shape[0]) * 1e-6
        cov_sqrt = linalg.sqrtm((fake_sigma_cpu + offset) @ (real_sigma_cpu + offset))
    if np.iscomplexobj(cov_sqrt): # 检查协方差矩阵的平方根是否包含复数
        cov_sqrt = cov_sqrt.real

    mean_norm = np.sum(diff_mu_cpu ** 2) # 此处不一样

    trace = np.trace(fake_sigma_cpu) + np.trace(real_sigma_cpu) - 2 * np.trace(cov_sqrt) 

    fid = mean_norm + trace
    del real_mu, real_sigma, fake_mu, fake_sigma    
    return fid

# 计算图像的熵
def calculate_entropy(image):
    hist = torch.histc(image, bins=256, min=0, max=1)
    hist = hist / torch.sum(hist)
    entropy = -torch.sum(hist * torch.log(hist + 1e-5))
    del hist
    return entropy.item()

# 计算Tenengrad (清晰度)
def calculate_tenengrad(image: torch.Tensor):
    """
    计算图像的 Tenengrad 值，用于评估图像的清晰度
    :param image: 形状为 [B, C, H, W] 的张量，代表 B 张 C 通道的图像
    :return: 每张图像的 Tenengrad 值
    """
    # Sobel 核心，用于计算梯度
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])  # 水平方向的 Sobel 核
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])  # 垂直方向的 Sobel 核
    
    # 将 Sobel 核转换为适用于批处理的形状 [1, 1, 3, 3]，并移动到 GPU（如果有的话）
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    # 将 Sobel 核扩展到与输入图像的通道数一致 [C, 1, 3, 3]，即 [3, 1, 3, 3]
    sobel_x = sobel_x.expand(image.shape[1], -1, -1, -1)  # [C, 1, 3, 3]
    sobel_y = sobel_y.expand(image.shape[1], -1, -1, -1)  # [C, 1, 3, 3]

    # 将 Sobel 核转换为适合进行卷积的形状 [C, 1, 3, 3]
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)
    
    # 计算水平和垂直梯度
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])  # [B, C, H, W]
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])  # [B, C, H, W]
    
    # 计算 Tenengrad 值 (每个像素的梯度平方和)
    grad_squared = grad_x**2 + grad_y**2
    
    # 计算每张图像的 Tenengrad 总值，可以选择对每个图像的所有像素求和或求均值
    tenengrad_values = grad_squared.view(image.shape[0], -1).sum(dim=1).mean().item()  # 对每张图像的所有像素求和
    del sobel_x, sobel_y, grad_x, grad_y
    return tenengrad_values/100000

# 计算VIF
def calculate_vif(image, reference):
    image = img_as_ubyte(image.cpu().detach().numpy()/ 255.0) 
    reference = img_as_ubyte(reference.cpu().numpy()/ 255.0) 
    # 简化版VIF计算
    vif = np.var(reference) / np.var(image) # a float64
    return vif

# 计算PSNR
def calculate_psnr(real_images, fake_images, max_pixel=255.0):
    """
    计算两组图像的PSNR值
    参数：
    real_images (torch.Tensor): 真实图像的张量，形状为 [N, C, H, W]
    fake_images (torch.Tensor): 生成图像的张量，形状为 [N, C, H, W]
    max_pixel (float): 图像像素的最大值（默认255.0，用于8位图像）

    返回：
    psnr (torch.Tensor): 每一对图像的PSNR值，形状为 [N]
    """
    # 确保图像张量在 [0, max_pixel] 的范围内
    mse = F.mse_loss(fake_images, real_images, reduction='none')  # 计算每个像素的均方误差
    mse = mse.mean(dim=(1, 2, 3))  # 对每张图像求均值，忽略通道维度

    # 计算PSNR
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)  # 根据公式计算PSNR
    psnr_mean = torch.mean(psnr)
    del mse

    return psnr_mean.item()

# 计算SSIM
def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11, sigma=1.5, device='cuda'):
    """
    计算两组图像的SSIM，输入是大小为 [8, 3, 1024, 1024] 的张量。
    假设输入图像的像素值范围是 [0, 255]，将其归一化到 [0, 1]。
    
    :param img1: 第一组图像，形状为 [8, 3, 1024, 1024]
    :param img2: 第二组图像，形状为 [8, 3, 1024, 1024]
    :param window_size: 用于计算局部均值和方差的窗口大小，通常为奇数
    :param sigma: 高斯窗口的标准差
    :param device: 'cuda' 或 'cpu'，指定计算设备
    
    :return: 每张图像的SSIM值的平均值，形状为 [8,]
    """
    # 将图像移到指定设备
    img1 = img1.to(device) / 255.0
    img2 = img2.to(device) / 255.0
    
    # 计算高斯窗口
    def gaussian_window(window_size, sigma, device):
        """生成高斯窗口并移动到设备"""
        _ = torch.arange(window_size).float().to(device)
        kernel_1d = torch.exp(-0.5 * ((_ - (window_size // 2)) / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d

    # 生成高斯窗口并移动到设备
    window = gaussian_window(window_size, sigma, device).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, window_size, window_size]
    
    # 为了适应多个通道，需要将窗口复制到每个通道
    window = window.expand(img1.shape[1], 1, window_size, window_size).contiguous()

    # 计算均值和方差
    def compute_mean_and_var(img):
        """计算图像的均值和方差"""
        mu = F.conv2d(img, window, padding=window_size // 2, groups=img.shape[1])
        mu_sq = mu ** 2
        sigma_sq = F.conv2d(img * img, window, padding=window_size // 2, groups=img.shape[1]) - mu_sq
        return mu, sigma_sq

    # 计算SSIM
    def compute_ssim_for_pair(x, y):
        """计算两张图像的SSIM"""
        mu_x, sigma_x_sq = compute_mean_and_var(x)
        mu_y, sigma_y_sq = compute_mean_and_var(y)
        
        # 计算协方差
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.shape[1]) - mu_x * mu_y
        
        # SSIM公式
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / denominator
        
        return ssim_map.mean()  # 返回SSIM的平均值
    
    ssim_values = []
    # 遍历每一张图片，计算对应的SSIM
    for i in range(img1.shape[0]):
        ssim_value = compute_ssim_for_pair(img1[i:i+1], img2[i:i+1])  # 只计算一张图片的SSIM
        ssim_values.append(ssim_value)
    
    return torch.tensor(ssim_values).mean().item()
