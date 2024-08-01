from bmgan_model import dense_unet_generator, ResNet_encoder, patch_discriminator
import sys
import os
import torch.nn as nn
import argparse

from monai_diffusion.generative.losses.adversarial_loss import PatchAdversarialLoss
from monai_diffusion.generative.losses.perceptual import PerceptualLoss
import sys
sys.path.append('/home1/yujiali/T1toPET')
from unet.utils.dataset import pair_PET_T1dataset
from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
import torch
import SimpleITK as sitk
import pdb

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12349'
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
    
    
    generator = dense_unet_generator().to(device)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    rec_loss = nn.L1Loss()

    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        epoch = ckpt['epoch']
    else:
        quit()
        
    output_folder = f'{args.exp_dir}/output/epoch={epoch}'
    os.makedirs(output_folder, exist_ok=True)

    
    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max={}, need_values=[])
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=1, 
                                                    sampler=eval_sampler,
                                                    drop_last=False,
                                                    num_workers=0)
    
    requires_grad(generator, False)
    generator.eval()
    eval_l1_loss = []
    
    for batch_idx, data in enumerate(eval_dataloader):
    
        t1_img, pet_img, info, subject, t1_date, pet_date = data
        t1_img, pet_img = t1_img.to(device), pet_img.to(device)

        random_vector = torch.randn([t1_img.shape[0], 8], device=device)
        with torch.no_grad():
            fake_pet = generator.module.forward(t1_img, random_vector)
        
        l1_loss_ = rec_loss(fake_pet, pet_img)
        eval_l1_loss.append(l1_loss_.item())

        if rank == 0 and batch_idx % 20 ==0:
            print('eval...', epoch, f'{batch_idx}/{len(eval_dataloader)}')
            
        rec_np = fake_pet[0].cpu().numpy().squeeze()
        os.makedirs(f'{output_folder}/{subject[0]}/{pet_date[0]}', exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(rec_np), f'{output_folder}/{subject[0]}/{pet_date[0]}/rec.nii.gz')
        
    print(f'epoch {epoch} eval l1 loss {np.mean(eval_l1_loss):.5f}')
    

    

def main():
    
    parser = argparse.ArgumentParser()
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[7], nargs='+')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='/home1/yujiali/T1toPET/bl_methods/BMGAN/exp')
    parser.add_argument('--seed', type=int, default=777)
    
    ## resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)
    
    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'

    args.PET_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'

    args.exp_dir = f'{args.exp_dir}/{args.pet_kind}'

    mp.spawn(test, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
    
    