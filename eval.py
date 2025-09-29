import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img,size=256):
    return F.interpolate(img, size=size)

def batch_generate(zs, netG, batch=8): 
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name): 
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str, help = 'Path to the checkpoint file')
    parser.add_argument('--artifacts', type=str, default=".", help='Output path')
    parser.add_argument('--cuda', type=int, default=0, help='GPU used')
    parser.add_argument('--batch', default=8, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=100, help = "The number of generated images")
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    parser.add_argument('--multiplier', type=int, default=10000, help='multiplier for model number')
    parser.add_argument('--epochs', type=str, default='0100', help='')
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)


    # 定义加载的模型号
    epoch = str(args.epochs)
    ckpt = f"model/path/{epoch}/train_results/test1/models/{epoch}.pth" #
    checkpoint = torch.load(ckpt, map_location=lambda a, b: a) 

    checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
    net_ig.load_state_dict(checkpoint['g'])


    # net_ig.eval()
    print('load checkpoint success, iteration %s'%epoch)

    net_ig.to(device)

    del checkpoint
    
    # Storage location of generated images
    dist = 'your/output/path/eval_%s'%(epoch)
    dist = os.path.join(dist, 'img')
    os.makedirs(dist, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.n_sample//args.batch)): # How many iterations are there in each epoch? This means looping through all iterations of a single epoch.
            noise = torch.randn(args.batch, noise_dim).to(device)
            g_imgs = net_ig(noise)[0]
            g_imgs = resize(g_imgs,args.im_size) # resize the image using given dimension
            for j, g_img in enumerate(g_imgs):
                vutils.save_image(
                    g_img.add(1).mul(0.5), 
                    os.path.join(dist, '%d.png'%(i*args.batch+j))
                    )#, normalize=True, range=(-1,1))
