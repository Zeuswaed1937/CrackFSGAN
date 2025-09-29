import torch
from torchvision import transforms
import numpy as np
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

from operation import ImageFolder, InfiniteSamplerWrapper
from mertics import calcute_lpips, calculate_fid, calculate_psnr, calculate_ssim, calculate_entropy, calculate_tenengrad, calculate_vif

import warnings
warnings.filterwarnings('ignore')

# 设置随机数种子，供数据集随机取样使用
random_seed = 114514
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def main(model_name, num):
    batch_size = 1000
    im_size = 256
    sample_size_r = 100
    sample_size_f = 101
    transform_list = [transforms.Resize((int(im_size),int(im_size))),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    trans = transforms.Compose(transform_list)


    data_root_r = '/root/datas/CE/Origin/'
    # sampler_r = InfiniteSamplerWrapper(SubsetRandomSampler(torch.arange(sample_size_r)))
    dataset_r = ImageFolder(root=data_root_r, transform=trans)
    dataloader_r = iter(DataLoader(dataset_r, batch_size=batch_size, shuffle=False,
                        num_workers=16, pin_memory=True))


    # data_root_f = '/root/autodl-tmp/results/ggp100'
    # data_root_f = '/root/datas/CE' + '/' + model_name + '/' + num + '/' + 'Crack/'
    data_root_f = '/root/autodl-tmp/results/ccic100' + model_name + '/eval_' + num + '/img'

    # sampler_f = SubsetRandomSampler(torch.arange(sample_size_f))
    dataset_f = ImageFolder(root=data_root_f, transform=trans)
    # print(len(dataset_f))
    dataloader_f = iter(DataLoader(dataset_f, batch_size=batch_size, shuffle=False,
                                    num_workers=16, pin_memory=True))
    # dataloader_f = iter(DataLoader(dataset_f, batch_size=batch_size, shuffle=False,
    #                     sampler = sampler_f, num_workers=4, pin_memory=True))


    device = torch.device("cuda:0")

    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.eval().to(device)

    fid_list = []
    lpips_list = []
    psnr_list = []
    ssim_list = []
    tenengrad_list = []
    vif_list = []
    entropy_list = []

    # 根据生成的图像迭代
    for iteration in tqdm(range(1, ((len(dataset_f) // batch_size) + 1))):
    # for iter in tqdm(range(1,2)):
        image_real = next(dataloader_r)
        image_fake = next(dataloader_f)

        lpips_value = calcute_lpips(image_real, image_fake, batch_size = batch_size)
        lpips_list.append(lpips_value)

        fid_value = calculate_fid(image_real, image_fake, inception_v3, device)
        fid_list.append(fid_value)

        psnr_value = calculate_psnr(image_real, image_fake, 255.0)
        psnr_list.append(psnr_value)

        ssim_value = calculate_ssim(image_real, image_fake,)
        ssim_list.append(ssim_value)

        tenengrad_value = calculate_tenengrad(image_fake)
        tenengrad_list.append(tenengrad_value)

        vif_value = calculate_vif(image_real, image_fake)
        vif_list.append(vif_value)
        
        entropy_value = calculate_entropy(image_fake)
        entropy_list.append(entropy_value)

    fid_values = sum(fid_list)/len(fid_list)
    lpips_values = sum(lpips_list)/len(lpips_list)
    psnr_values = sum(psnr_list)/len(psnr_list)
    ssim_values = sum(ssim_list)/len(ssim_list)
    tenengrad_values = sum(tenengrad_list)/len(tenengrad_list)
    vif_values = sum(vif_list)/len(vif_list)
    entropy_values = sum(entropy_list)/len(entropy_list)

    print(f'{model_name} with {num} fid_value:{fid_values:.5f}')
    print(f'{model_name} with {num} lpips_value:{lpips_values:.5f}')
    print(f'{model_name} with {num} psnr_value:{psnr_values:.5f}')
    print(f'{model_name} with {num} ssim_value:{ssim_values:.5f}')
    print(f'{model_name} with {num} tenengrad_value:{tenengrad_values:.5f}')
    print(f'{model_name} with {num} vif_value:{vif_values:.5f}')
    print(f'{model_name} with {num} entropy_alue:{entropy_values:.5f}')

if __name__ == "__main__":
    # model_names = ['BigGAN256', 'DCGAN', 'FastGAN', 'GAN-GP', 'InfoGAN', 'SD-L', 'SN-GAN', 'WGAN']
    # nums = ['100', '20k']
    model_names = ['notA', 'notD', 'notG', 'notN']
    nums = ['2100', '1800']
    # model_names = ['FastGAN']
    # nums = ['100s']
    for num in nums:
        for model_name in model_names:
        
            main(model_name, num)