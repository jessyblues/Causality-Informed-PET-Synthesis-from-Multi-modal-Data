import torch
import os

import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
import sys
sys.path.append('/home1/yujiali/T1toPET')
sys.path.append('/home1/yujiali/diffusion')
import pdb

from unet.utils.dataset import pair_PET_T1dataset
import argparse
from torch.backends import cudnn
import torch.distributed as dist
import numpy as np
import random
import torch.multiprocessing as mp
from monai_diffusion.generative.networks.nets.atten_unet import AttenUNet
import SimpleITK as sitk

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch.nn.functional as F
import pickle
import ants

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12245'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
         

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
    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)
        
    model_dict['atten_unet_def']['cross_attention_dim'] = len(need_values)
    atten_unet_dict = model_dict['atten_unet_def']
    unet = AttenUNet(**atten_unet_dict)
    unet.to(device)
    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    

    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        unet.load_state_dict(ckpt['unet'])
        begin_epoch = ckpt['epoch']
    else:
        begin_epoch = 0
    
    sample_folder=os.path.join(args.exp_dir, 'test_output', f'epoch={begin_epoch}')
    os.makedirs(sample_folder, exist_ok=True)
    
    
    l1_loss = nn.L1Loss()
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=5, sigma=0.5).to(device)
  
    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                    PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=args.batch_size, 
                                                    sampler=eval_sampler,
                                                    drop_last=False,
                                                    num_workers=0)
    
    unet.eval()
    metrics = {}
    
    for k in args.metrics:
        metrics[k] = []
    for batch_idx, batch in enumerate(eval_dataloader):
        
        t1_img, pet_img, info, subject, t1_date, pet_date = batch
        
        t1_img, pet_img = t1_img.to(device), pet_img.to(device)
        date_folder = os.path.join(sample_folder, subject[0], pet_date[0])
        
        if args.use_condition:
            condition = info.to(device).view(pet_img.shape[0], 1, len(need_values))
        else:
            condition = torch.zeros((pet_img.shape[0], 1, len(need_values)), device=device)
            
        os.makedirs(date_folder, exist_ok=True)
        
        with torch.no_grad():
            output_pet = unet(t1_img, condition)
        
        pet_img = pet_img.squeeze()
        #pet_img = (pet_img - torch.min(pet_img)) / (torch.max(pet_img) - torch.min(pet_img))
        output_pet = output_pet.squeeze()
        #output_pet = torch.clip(output_pet-output_pet[0, 0, 0], min=0)
        #output_pet = output_pet * (pet_img>0)
        #output_pet = output_pet/torch.mean(output_pet)*torch.mean(pet_img)
        #output_pet = (output_pet - torch.min(output_pet)) / (torch.max(output_pet) - torch.min(output_pet))
        
        img_np = pet_img.cpu().numpy().squeeze()
        rec_np = output_pet.cpu().numpy().squeeze()
        
        mask = ants.get_mask(ants.from_numpy(img_np))
        rec_np = rec_np*(mask.numpy())
        
        if 'MAE' in args.metrics:
            #pdb.set_trace()
            metrics['MAE'].append(torch.nn.L1Loss()(output_pet, pet_img).item())
            print(metrics['MAE'][-1])
        if 'SSIM' in args.metrics:
            ssim = ms_ssim(output_pet.unsqueeze(0).unsqueeze(0), pet_img.unsqueeze(0).unsqueeze(0))
            #print(ssim)
            #quit()
            if not np.isnan(ssim):
                metrics['SSIM'].append(ssim.item())
            #print(ssim.item())
        if 'PSNR' in args.metrics:
            metrics['PSNR'].append(10*torch.log10(1/F.mse_loss(pet_img, output_pet)).item()) 
        
        sitk.WriteImage(sitk.GetImageFromArray(img_np), os.path.join(date_folder, 'ori.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(rec_np), os.path.join(date_folder, 'rec.nii.gz'))
        
        
        
    print(ckpt['epoch'], ckpt['eval_loss'])
    for k ,v in metrics.items():
        print(k, np.mean(v), np.std(v))


def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--model_config_path', type=str, 
                        default='./unet/config/training.json')
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[0], nargs='+')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='./unet/exp')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--metrics', default=['MAE', 'SSIM', 'PSNR'], type=str, nargs='+')
    parser.add_argument('--print_image', default=False, action='store_true')
    parser.add_argument('--use_condition', default=False, action='store_true')
    
    ## resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)
    
    args.training_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_training_with_csf.csv'
    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'
    args.PET_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'
    args.exp_dir = f'{args.exp_dir}/conditional/{args.pet_kind}' if args.use_condition else \
        f'{args.exp_dir}/unconditional/{args.pet_kind}'
    args.min_and_max = f'/home1/yujiali/T1toPET/unet/config/{args.pet_kind}_min_and_max.pkl'
    
    mp.spawn(test, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
