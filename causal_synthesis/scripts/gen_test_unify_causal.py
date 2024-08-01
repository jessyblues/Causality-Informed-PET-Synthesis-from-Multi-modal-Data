import os
import json
import sys
sys.path.append('/home1/yujiali/diffusion')
from monai_diffusion.generative.networks.nets.autoencoderkl import AutoencoderKL
from monai_diffusion.generative.networks.nets.atten_decoder import DiffusionModelDecoder

sys.path.append('/home1/yujiali/T1toPET')

from unet.utils.dataset import pair_PET_T1dataset

from torch.backends import cudnn
import torch.distributed as dist
import random

from torch.utils.data import DataLoader
import torch
import SimpleITK as sitk
import pdb
import numpy as np
import argparse
import torch.multiprocessing as mp
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.fft import fft
from monai_diffusion.generative.losses.perceptual import PerceptualLoss
from monai_diffusion.generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai_diffusion.generative.losses.adversarial_loss import PatchAdversarialLoss
import pickle
import ants

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '22345'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def requires_grad(net, flag):
    
    for k,v in net.named_parameters():
        v.requires_grad=flag

    return net

    
def test(rank, args):
    
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)

    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda", args.cuda_ids[rank])
    
    
    
    with open(args.model_config_path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
        
    need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.pet_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']

    ## create and load vae
   # print()
    t1_autoencoder_dict = model_dict['t1_autoencoder_def']
    t1_autoencoder = AutoencoderKL(**t1_autoencoder_dict)
    t1_autoencoder.to(device)
    
    pet_decoder_dict = model_dict['pet_decoder_def']
    pet_decoder_dict['cross_attention_dim'] = len(need_values)
    pet_decoder = DiffusionModelDecoder(**pet_decoder_dict)
    pet_decoder.to(device)
    
    t1_autoencoder = torch.nn.parallel.DistributedDataParallel(t1_autoencoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    pet_decoder = torch.nn.parallel.DistributedDataParallel(pet_decoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)

    
    
    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        pet_decoder.load_state_dict(ckpt['pet_decoder'])
        t1_autoencoder.load_state_dict(ckpt['t1_autoencoder'])
        epoch = ckpt['epoch']
    else:
        quit()
        
    output_folder=os.path.join(args.exp_dir, 'test_output', f'epoch={epoch}')
    os.makedirs(output_folder, exist_ok=True)
    
    need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.pet_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']
    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)


    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
                                          #random_crop=True, random_crop_size=(48, 64, 48))
    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=1, 
                                                    sampler=eval_sampler,
                                                    drop_last=False,
                                                    num_workers=0)


            
    pet_decoder.eval()
    t1_autoencoder.eval()
    
    
    
    for batch_idx, batch in enumerate(eval_dataloader):
        
        
        t1_img, pet_img, info, subject, t1_date, pet_date = batch
        t1_img, pet_img = t1_img.to(device), pet_img.to(device)
        info = info.view(t1_img.shape[0], 1, -1).to(device)
        
        with torch.no_grad():
            t1_rec, z_mu, z_sigma = t1_autoencoder(t1_img)       
            eps_PET = torch.randn_like(z_sigma)
            reconstruction = pet_decoder(z_mu+eps_PET*z_sigma, info)
        

        output_folder_ = f'{output_folder}/{subject[0]}/{pet_date[0]}'
        os.makedirs(output_folder_, exist_ok=True)
        #img_np = pet_img[0].cpu().numpy().squeeze()
        
        reconstruction = (reconstruction - torch.min(reconstruction)) / (torch.max(reconstruction) - torch.min(reconstruction))
        rec_np = reconstruction.cpu().numpy().squeeze()
        mask = ants.get_mask(ants.from_numpy(rec_np))
        rec_np = rec_np*(mask.numpy())
        
        sitk.WriteImage(sitk.GetImageFromArray(rec_np), f'{output_folder_}/rec.nii.gz')
        #sitk.WriteImage(sitk.GetImageFromArray(rec_np), os.path.join(sample_folder, 'epoch={}_rec.nii.gz'.format(epoch)))
        print(f'{batch_idx}/{len(eval_dataloader)} finished!')
        


    
def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--model_config_path', type=str, 
                        default='/home1/yujiali/T1toPET/causal_synthesis/configs/training_unify_causal.json')
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[0], nargs='+')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='/home1/yujiali/T1toPET/causal_synthesis/exp_causal')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--min_and_max', type=str, default='/home1/yujiali/T1toPET/unet/config/AV45_min_and_max.pkl')
    
    ## pretrained model
    parser.add_argument('--resume', type=str, default=None)
    
    ## dataset
    
    args = parser.parse_args()
    
    args.training_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_training_with_csf.csv'
    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'
    
    args.PET_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'

    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)

    mp.spawn(test, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
    
    