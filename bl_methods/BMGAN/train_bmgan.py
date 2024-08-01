
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

def kl_divergence(mu, logvar):
    """
    计算高斯分布与标准正态分布之间的 KL 散度
    :param mu: 高斯分布的均值
    :param logvar: 高斯分布的对数方差
    :return: KL 散度
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def train(rank, args):
    
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)

    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda", args.cuda_ids[rank])
    
    writer_folder=os.path.join(args.exp_dir, 'log')
    ckpt_folder=os.path.join(args.exp_dir, 'ckpt')
    sample_folder=os.path.join(args.exp_dir, 'visual')
    
    os.makedirs(writer_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(sample_folder, exist_ok=True)
    
    generator = dense_unet_generator().to(device)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    discriminator = patch_discriminator().to(device)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    encoder = ResNet_encoder().to(device)
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True).to(device)
    rec_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=3).to(device)
    
    
    g_optimizer = torch.optim.Adam(
        [
            {"params": generator.parameters()}     
        ],
        lr = 2e-4
        )
    d_optimizer = torch.optim.Adam(
        [
            {"params": discriminator.parameters()}         
        ],
        lr = 2e-4
        )
    e_optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters()}         
        ],
        lr = 2e-4
        )

    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        encoder.load_state_dict(ckpt['encoder'])
        begin_epoch = ckpt['epoch']+1
    else:
        begin_epoch = 0
        ckpt = None
    
    if ckpt is not None and 'g_optimizer' in ckpt.keys():
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        e_optimizer.load_state_dict(ckpt['e_optimizer'])
    
    training_dataset = pair_PET_T1dataset(info_csv=args.training_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max={}, need_values=[])
    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max={}, need_values=[])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset,  
                                                    batch_size=args.batch_size, 
                                                    sampler=train_sampler,
                                                    drop_last=True,
                                                    num_workers=0)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=1, 
                                                    sampler=eval_sampler,
                                                    drop_last=False,
                                                    num_workers=0)
    
    best_eval_loss = 100
    for epoch in range(begin_epoch, args.epochs):        
        
        
        train_sampler.set_epoch(epoch)
        
        discriminator.train()
        encoder.train()
        generator.train()
        
        for batch_idx, data in enumerate(training_dataloader):
            
            #break
            requires_grad(discriminator, False)
            requires_grad(encoder, False)
            requires_grad(generator, True)
            
            t1_img, pet_img, info, subject, t1_date, pet_date = data
            t1_img, pet_img = t1_img.to(device), pet_img.to(device)

            random_vector = torch.randn([t1_img.shape[0], 8], device=device)
            fake_pet = generator(t1_img, random_vector)
            
            logits_fake = discriminator(fake_pet.contiguous().float())[-1]
            adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            l1_loss_ = rec_loss(fake_pet, pet_img)
            perceputal_loss_ = perceptual_loss(fake_pet, pet_img)
            
            generator_loss = adv_loss_ + args.lamda_l1*l1_loss_ + args.lamda_preceputal*perceputal_loss_
            
            g_optimizer.zero_grad()
            generator_loss.backward()
            g_optimizer.step()

            requires_grad(discriminator, False)
            requires_grad(encoder, True)
            requires_grad(generator, False)        
            
            #print(pet_img.shape)
            with torch.no_grad():
                fake_pet = generator(t1_img, random_vector)
            real_pet_encoded_mu, real_pet_encoded_log_var = encoder(pet_img)
            fake_pet_encoded_mu, fake_pet_encoded_log_var = encoder(fake_pet.contiguous().float())
            
            
            kl_divergence_loss = kl_divergence(real_pet_encoded_mu, real_pet_encoded_log_var) + \
                                kl_divergence(fake_pet_encoded_mu, fake_pet_encoded_log_var)
            kl_divergence_loss = kl_divergence_loss.mean()
                
            e_optimizer.zero_grad()
            kl_divergence_loss.backward()
            e_optimizer.step()
            
            
            requires_grad(discriminator, True)
            requires_grad(encoder, False)
            requires_grad(generator, False)   
            
            
            with torch.no_grad():
                fake_pet = generator(t1_img, random_vector)
            
            logits_fake = discriminator(fake_pet.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            loss_d_fake.backward()
            logits_real = discriminator(pet_img.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d_real.backward()
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            
            d_loss = discriminator_loss
            d_loss = d_loss.mean()
            
            if batch_idx % 20 == 0 and rank==0:
                print(f'epoch {epoch} batch{batch_idx}/{len(training_dataloader)} train l1 loss {l1_loss_.item():.5f}'+ 
                f'gen loss {generator_loss.item():.5f}'+
                f'encoder loss {kl_divergence_loss.item():.5f} discriminator loss {discriminator_loss.item():.5f}')
            
        if epoch % args.eval_every == 0 and rank == 0:
            
            requires_grad(discriminator, False)
            requires_grad(encoder, False)
            requires_grad(generator, False)
            discriminator.eval()
            encoder.eval()
            generator.eval()
            eval_generator_loss = []
            eval_encoder_loss = []
            eval_dis_loss = []
            eval_l1_loss = []
            
            for batch_idx, data in enumerate(eval_dataloader):
            
                t1_img, pet_img, info, subject, t1_date, pet_date = data
                t1_img, pet_img = t1_img.to(device), pet_img.to(device)

                random_vector = torch.randn([t1_img.shape[0], 8], device=device)
                with torch.no_grad():
                    fake_pet = generator.module.forward(t1_img, random_vector)
                    logits_fake = discriminator.module.forward(fake_pet.contiguous().float())[-1]
                #print('gen finished!')
                adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                
                l1_loss_ = rec_loss(fake_pet, pet_img)
                perceputal_loss_ = perceptual_loss(fake_pet, pet_img)
                
                generator_loss = adv_loss_ + args.lamda_l1*l1_loss_ + args.lamda_preceputal*perceputal_loss_
                
                eval_generator_loss.append(generator_loss.item())
                eval_l1_loss.append(l1_loss_.item())
                #print('gen loss finished!')
                with torch.no_grad():
                    real_pet_encoded_mu, real_pet_encoded_log_var = encoder.module.forward(pet_img)
                    fake_pet_encoded_mu, fake_pet_encoded_log_var = encoder.module.forward(fake_pet.contiguous().float())
                
                kl_divergence_loss = kl_divergence(real_pet_encoded_mu, real_pet_encoded_log_var) + \
                                    kl_divergence(fake_pet_encoded_mu, fake_pet_encoded_log_var)
                
                #print('encoder loss finished!')               
                eval_encoder_loss.append(kl_divergence_loss.item())
                                    

                with torch.no_grad():
                    logits_fake = discriminator.module.forward(fake_pet.contiguous().detach())[-1]
                    logits_real = discriminator.module.forward(pet_img.contiguous().detach())[-1]
                    
                #print('dis loss finished!')    
                #pdb.set_trace()
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                #print('a finished!') 
                d_loss = discriminator_loss
                d_loss = d_loss.mean()
                #print('b finished!') 
                eval_dis_loss.append(d_loss.item())
                #pdb.set_trace()
                if rank == 0 and batch_idx % 20 ==0:
                    print('eval...', epoch, f'{batch_idx}/{len(eval_dataloader)}')
                
            print(f'epoch {epoch} eval l1 loss {np.mean(eval_l1_loss):.5f} gen loss {np.mean(eval_generator_loss):.5f}'+
                f'eval encoder loss {np.mean(eval_encoder_loss):.5f}')
            
            img_np = pet_img[0].cpu().numpy().squeeze()
            rec_np = fake_pet[0].cpu().numpy().squeeze()
            t1_np = t1_img[0].cpu().numpy().squeeze()
            
            sitk.WriteImage(sitk.GetImageFromArray(img_np), os.path.join(sample_folder, 'epoch={}_ori.nii.gz'.format(epoch)))
            sitk.WriteImage(sitk.GetImageFromArray(rec_np), os.path.join(sample_folder, 'epoch={}_rec.nii.gz'.format(epoch)))
            sitk.WriteImage(sitk.GetImageFromArray(t1_np), os.path.join(sample_folder, 'epoch={}_t1.nii.gz'.format(epoch)))
            
            if best_eval_loss > np.mean(eval_l1_loss):
                
                state_dict = {
                    'generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'encoder':encoder.state_dict(),
                    'epoch':epoch,
                    'l1_loss':np.mean(eval_l1_loss)
                }
                
                torch.save(state_dict, f'{ckpt_folder}/best.ckpt')
                best_eval_loss = np.mean(eval_l1_loss)
        
        if epoch % args.save_every == 0 and rank == 0:
            
            state_dict = {
                'generator':generator.state_dict(),
                'discriminator':discriminator.state_dict(),
                'encoder':encoder.state_dict(),
                'epoch':epoch,
                'd_optimizer':d_optimizer.state_dict(),
                'e_optimizer':e_optimizer.state_dict(),
                'g_optimizer':g_optimizer.state_dict()
            }
            
            torch.save(state_dict, f'{ckpt_folder}/epoch={epoch}.ckpt')
            best_eval_loss = np.mean(eval_l1_loss)
    

def main():
    
    parser = argparse.ArgumentParser()
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[0, 1, 2, 3], nargs='+')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='/home1/yujiali/T1toPET/bl_methods/BMGAN/exp')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
 
    parser.add_argument('--lamda_l1', type=float, default=20)
    parser.add_argument('--lamda_preceputal', type=float, default=8)
    
    ## resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)
    
    args.training_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_training_with_csf.csv'
    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'

    args.PET_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'

    args.exp_dir = f'{args.exp_dir}/{args.pet_kind}'

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
    
    