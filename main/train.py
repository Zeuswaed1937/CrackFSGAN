import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import utils as vutils

import numpy as np
from skimage import img_as_ubyte
from torchvision import models, transforms
from mertics import calcute_lpips, calculate_fid, calculate_psnr, calculate_ssim, calculate_entropy, calculate_tenengrad, calculate_vif
import argparse
import random
from tqdm import tqdm

from scipy import linalg
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from concurrent.futures import ThreadPoolExecutor


policy = 'color,translation'

import LPIPS_calculation
percept = LPIPS_calculation.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True) # Load VGG as feature extractor


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    # If it is a real image
    if label=="real":
        part = random.randint(0, 3)
        # The pred here is averaged to represent the discriminator loss
        # Returns tensors and lists of tensors
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)

        # After the err calculation is completed, it is back-propagated and all used.
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() + \
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() + \
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    # If it is not a real image, only processing the pred result
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

import os

# Function to create log file
def initialize_log_file(log_path):
    if not os.path.exists(log_path):  # 
        with open(log_path, "w") as log_file:
            # Write header or initial information
            log_file.write("Iteration, LPIPS, FID, PSNR, SSIM, Entropy, Tenengrad, VIF\n")
            print(f"Log file created at: {log_path}")
    else:
        print(f"Log file exists at: {log_path}")



def train(args):

    data_root = args.path # The path of the dataset
    total_epochs = args.epoch
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    metric_sup = args.metric_sup # Whether to generate indicators on an by-iteration basis

    ndf = 64 # The number of feature maps
    ngf = 64
    nz = 256
    nlr = 0.0002 # Learning rate
    nbeta1 = 0.5

    use_cuda = True 
    multi_gpu = True 
    
    saved_model_folder, saved_image_folder = get_dir(args) 

    save_num_ckpts = args.save_num_ckpts # The number of models you plan to store
    sample_size = args.sample_size  # Add new parameters to control the number of randomly selected images
    random_seed = args.random_seed  # Ensure reproducible results for random image sampling
    metric_yep = args.metric_yep

    iteration_per_epoch = sample_size // batch_size # Iterations required for each epoch
    total_iterations = total_epochs * iteration_per_epoch # The total number of iterations required, see the progress bar
    save_interval = total_iterations // save_num_ckpts # Model saving frequency, calculated by iteration

    # Set the random number seed for random sampling of the data set
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
        # Create an infinite sampling decorator based on the subset random sampler
        sampler = InfiniteSamplerWrapper(SubsetRandomSampler(torch.arange(sample_size)))
    else:
        # All data is directly sampled using wireless decorator
        sampler = InfiniteSamplerWrapper(dataset)

    # Officially load data from here
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=sampler, num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    #From models.py import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None': 
        ckpt = torch.load(checkpoint) 
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    # Create a log file to record metrics
    # Log file path
    log_path = os.path.join(args.output_path, "metrics_log.txt")
    # Initialize log file
    initialize_log_file(log_path)
    metrics_log_file = open(log_path, "a")

    epoch_metrics = {
        "lpips": [], # the small the better
        "fid": [], # the small the better
        "psnr": [],
        "ssim": [],
        "entropy": [],
        "tenengrad": [],
        "vif": []
    }

    epoch_losses = {
        "d_losses": [],  # Discriminator loss, directly output from the function result
        "g_losses": []   # Generator loss, directly output from the function result
    }
    
    current_epoch = 0 # Specify the starting iteration number

    # Used to calculate the FID indicator. 
    # This is different from the original calculation method and uses the same Ic model.
    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.eval().to(device)
    metrics_log_file = open(log_path, "a")

    # Each iteration of the loop begins
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)
        fake_image = fake_images[0]

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## Train Discriminator
        netD.zero_grad()

        # The discriminator loss is calculated here
        # The return value of D is a single tensor or a list of tensors.
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        # For non-real images, D returns only a single tensor.
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        epoch_losses["d_losses"].append(err_dr)


        ## Train Generator
        netG.zero_grad()
        # The Generator loss is calculated here
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        epoch_losses["g_losses"].append(err_g.item())

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)
      
        
        if (iteration + 1) % metric_yep == 0: # This is to optimize the operation speed and use the approximate average loss calculation
            with torch.no_grad():
                if metric_sup == True:  
                    with ThreadPoolExecutor() as executor: # Using asynchronous calling scheme
                        # Calculate LPIPS
                        lpips_value = executor.submit(calcute_lpips, real_image, fake_images[0], batch_size).result()
                        # Calculate FID pass, the main computational cost is on FID output
                        fid_value = executor.submit(calculate_fid, real_image, fake_image, inception_v3, device).result()              
                        # Calculate PSNR fuck,oom
                        psnr_value = calculate_psnr(fake_image, real_image)
                        # Calculate SSIM pass
                        ssim_value = calculate_ssim(fake_image, real_image)
                        # CalculateEntropy pass
                        entropy_value = calculate_entropy(fake_image)
                        # Calculate Tenengrad
                        tenengrad_value = calculate_tenengrad(fake_image)
                        # Calculate VIF pass, but make train slow
                        vif_value = calculate_vif(fake_image, real_image)
                else: # # Do not supervise the generation of any indicators
                    lpips_value = 0
                    fid_value = 0
                    psnr_value = 0
                    ssim_value = 0
                    entropy_value = 0
                    tenengrad_value = 0
                    vif_value = 0

            # Cumulative index value
            epoch_metrics["lpips"].append(lpips_value)
            epoch_metrics["fid"].append(fid_value)
            epoch_metrics["psnr"].append(psnr_value)
            epoch_metrics["ssim"].append(ssim_value)
            epoch_metrics["entropy"].append(entropy_value)
            epoch_metrics["tenengrad"].append(tenengrad_value)
            epoch_metrics["vif"].append(vif_value)

        # Determine whether an epoch is completed
        if (iteration + 1) % iteration_per_epoch == 0:
            current_epoch += 1

            # Calculate the average value for each indicator
            avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()} # Calculate the approximate average loss for each iteration
            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()} # Calculate the average loss for each iteration
            print(f"Epoch {current_epoch} Metrics:")

            for metric, value in avg_metrics.items(): # Output the values ​​of each indicator for each iteration
                print(f"{metric}: {value:.4f}")
            for loss_type, avg_loss in avg_losses.items(): # Output the average loss for each iteration
                print(f"{loss_type}: {avg_loss:.4f}")
            
            # Save to log file
            metrics_log_file.write(f"Epoch {current_epoch}:, "
                                   f"LPIPS: {avg_metrics['lpips']:.4f}, FID: {avg_metrics['fid']:.4f}, "
                                   f"PSNR: {avg_metrics['psnr']:.4f}, SSIM: {avg_metrics['ssim']:.4f}, "
                                   f"Entropy: {avg_metrics['entropy']:.4f}, Tenengrad: {avg_metrics['tenengrad']:.4f}, "
                                   f"VIF: {avg_metrics['vif']:.4f}, D Loss: {avg_losses['d_losses']:.4f}, "
                                   f"G Loss: {avg_losses['g_losses']:.4f}\n")

            metrics_log_file.flush()

            # Reset indicator accumulation
            epoch_metrics = {k: [] for k in epoch_metrics}
            epoch_losses = {k: [] for k in epoch_losses}

        save_model_epoch = iteration // iteration_per_epoch
        # Save test sample images, 8 in 1
        if iteration % (save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), 
                                  saved_image_folder+'/%d.jpg'%iteration, 
                                  nrow=10)
                vutils.save_image(torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), 
                        saved_image_folder+'/rec_%d.jpg'%iteration)
            load_params(netG, backup_para)
        
        # Save the test model .pth file
        if iteration % (save_interval) == 0 or iteration == total_iterations:
            
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict()}, 
                        saved_model_folder+'/%d.pth'%iteration)
            
    metrics_log_file.close()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='your/dataset/path', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='your/output/path', help='Output path for the train results') 
    parser.add_argument('--sample_size', type=int, default=100, help='Number of random samples to use from the dataset.')
    parser.add_argument('--metric_yep', type = int, default = 1, help= "Calculate the sampling density of each indicator")
    parser.add_argument('--save_num_ckpts', type=int, default=50, help='number of save model you want')
    parser.add_argument('--epoch', type = int, default = 5000, help= "Total Epoch amounts")
    parser.add_argument('--im_size', type=int, default=512, help='Adjust image resolution')
    parser.add_argument('--metric_sup', type = bool, default = False, help = 'Whether to output evaluation indicators')

    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name') 
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch number of images')
    
    parser.add_argument('--ckpt', type=str, default='None', help='If it exists, it is the path of the checkpoint weight and is related to continued training')
    parser.add_argument('--workers', type=int, default=8, help='Loading training data')
    parser.add_argument('--random_seed', type=int, default=114514, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    print(args)

    train(args)
