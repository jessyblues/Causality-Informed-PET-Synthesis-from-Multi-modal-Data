import os
import json
import sys
sys.path.append('.')
from monai_diffusion.generative.networks.nets.atten_unet import DiffusionModelEncoder
from monai_diffusion.generative.networks.nets.atten_decoder import DiffusionModelDecoder
from monai_diffusion.generative.networks.nets.autoencoderkl import Decoder

sys.path.append('./T1toPET')

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
from monai_diffusion.generative.losses.perceptual import PerceptualLoss
from monai_diffusion.generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai_diffusion.generative.losses.adversarial_loss import PatchAdversarialLoss
import pickle


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def frozen_layer(net, trainable_layer_idx, frozen_encoder=True, frozen_decoder=True):
        
        
    for k,v in net.named_parameters():
        if True in ['encoder.blocks.{}.'.format(trainable_layer_idx_) in k for trainable_layer_idx_ in trainable_layer_idx] or not frozen_encoder:
            v.requires_grad=True
        elif True in ['decoder.blocks.{}.'.format(trainable_layer_idx_) in k for trainable_layer_idx_ in trainable_layer_idx] or not frozen_decoder:
            v.requires_grad=True
        else:
            v.requires_grad=False
    
    return net

def requires_grad(net, flag):
    
    for k,v in net.named_parameters():
        v.requires_grad=flag

    return net

def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence between the learned latent distribution
    and the standard normal distribution.

    Parameters:
    mu (torch.Tensor): Mean of the latent Gaussian distribution
    logvar (torch.Tensor): Log variance of the latent Gaussian distribution

    Returns:
    torch.Tensor: The KL divergence loss
    """
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by batch size
    kl_loss /= mu.size(0)
    return kl_loss
    
    
    
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
    
    
    if rank == 0:
        writer = SummaryWriter(log_dir=writer_folder, flush_secs=10)
    
    with open(args.model_config_path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
        
    need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.pet_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']

    ## create and load vae
   # print()
    t1_autoencoder_dict = model_dict['t1_autoencoder_def']
    t1_encoder = DiffusionModelEncoder(**t1_autoencoder_dict['atten_encoder'])
    t1_decoder = Decoder(**t1_autoencoder_dict['decoder'])
    t1_encoder.to(device)
    t1_decoder.to(device)
    
    pet_decoder_dict = model_dict['pet_decoder_def']
    pet_decoder_dict['cross_attention_dim'] = len(need_values)
    pet_decoder = DiffusionModelDecoder(**pet_decoder_dict)
    pet_decoder.to(device)
    
    t1_encoder = torch.nn.parallel.DistributedDataParallel(t1_encoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    t1_decoder = torch.nn.parallel.DistributedDataParallel(t1_decoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    pet_decoder = torch.nn.parallel.DistributedDataParallel(pet_decoder, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    discriminator = PatchDiscriminator(**model_dict['discriminator']).to(device)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    perceptual_loss = PerceptualLoss(**model_dict['perceptual_network']).to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True).to(device)
    
    training_config = model_dict['training']
    base_lr = training_config["base_lr"]
    disc_lr = training_config["disc_lr"]
    perceptual_weight = training_config["perceptual_weight"]
    adv_weight = training_config["adv_weight"]
    kl_weight = training_config["kl_weight"]
    
    
    g_optimizer = Adam(
        [
            {"params": pet_decoder.parameters()},
            {"params": t1_encoder.parameters()},   
            {"params": t1_decoder.parameters()},  
        ],
        lr = base_lr
        )

    
    d_optimizer = Adam(
        [
            {"params": discriminator.parameters()}         
        ],
        lr = disc_lr
        )
    
    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        pet_decoder.load_state_dict(ckpt['pet_decoder'])
        discriminator.load_state_dict(ckpt['discriminator'])
        t1_encoder.load_state_dict(ckpt['t1_encoder'])
        t1_decoder.load_state_dict(ckpt['t1_decoder'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        begin_epoch = ckpt['epoch']+1
    else:
        begin_epoch = 0
    
    l1_loss = nn.L1Loss()
    
    need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.pet_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']
    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)

    
    training_dataset = pair_PET_T1dataset(info_csv=args.training_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
                                          #random_crop=True, random_crop_size=(48, 64, 48))
    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
                                          #random_crop=True, random_crop_size=(48, 64, 48))
    
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
    
    for epoch in range(begin_epoch, args.epochs):

        pet_decoder.train()
        discriminator.train()
        t1_encoder.train()
        t1_decoder.train()
        training_dataloader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(training_dataloader):
            
            
            
            requires_grad(discriminator, False)
            requires_grad(pet_decoder, True)
            requires_grad(t1_encoder, True)
            requires_grad(t1_decoder, True)
            
            t1_img, pet_img, info, subject, t1_date, pet_date = data
            t1_img, pet_img = t1_img.to(device), pet_img.to(device)

            latent = t1_encoder(t1_img)
            z_mu = latent[:, :3]
            z_sigma = latent[:, 3:]
            
            eps_T1 = torch.randn_like(z_sigma)
            t1_rec = t1_decoder(z_mu+eps_T1*z_sigma)
            
            info = info.view(t1_img.shape[0], 1, -1).to(device)
            
            eps_PET = torch.randn_like(z_sigma)

            rec_pet = pet_decoder(z_mu+eps_PET*z_sigma, info)

            
            rec_loss_t1 = l1_loss(t1_rec, t1_img)
            kl_loss_t1 = kl_divergence(z_mu, z_sigma)
            
            rec_loss_pet = l1_loss(rec_pet, pet_img)
            p_loss = perceptual_loss(rec_pet.float(), pet_img.float())

            if adv_weight > 0:
                logits_fake = discriminator(rec_pet.contiguous().float())[-1]
                #print(logits_fake.shape)
                adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                adv_loss_ = torch.tensor([0.0]).to(device)

            g_loss = rec_loss_pet  + perceptual_weight * p_loss + adv_weight * adv_loss_
            t1_loss = rec_loss_t1 + kl_weight*kl_loss_t1
            
            g_loss = g_loss.mean()
            rec_loss_pet = rec_loss_pet.mean()
            p_loss = p_loss.mean()
            adv_loss_ = adv_loss_.mean()
            t1_loss = t1_loss.mean()
            
            g_optimizer.zero_grad()
            (g_loss+t1_loss).backward()
            #t1_loss.backward()
            g_optimizer.step()
            
            ## training d
            if adv_weight > 0:
                
                requires_grad(discriminator, True)
                requires_grad(pet_decoder, False)
                requires_grad(t1_encoder, False)
                requires_grad(t1_decoder, False)
                d_optimizer.zero_grad()       
                
                t1_img, pet_img, info, subject, t1_date, pet_date = data
                t1_img, pet_img = t1_img.to(device), pet_img.to(device)
                info = info.view(t1_img.shape[0], 1, -1).to(device)

                #with torch.no_grad():
                latent = t1_encoder(t1_img)
                z_mu = latent[:, :3]
                z_sigma = latent[:, 3:]
                   
                eps_PET = torch.randn_like(z_sigma)
                rec_pet = pet_decoder(z_mu+eps_PET*z_sigma, info)
                
                #print(logits_fake)
                logits_fake = discriminator(rec_pet.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_fake.backward()
                logits_real = discriminator(pet_img.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d_real.backward()
                #print(rank)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                

                d_loss = discriminator_loss
                d_loss = d_loss.mean()
                
                #d_loss.backward()
                d_optimizer.step()
            
            if rank == 0:
                steps = epoch*len(training_dataloader) + batch_idx
                writer.add_scalar("trainig/rec_pet_loss", rec_loss_pet.item(), steps)
                writer.add_scalar("trainig/rec_t1_loss", rec_loss_t1.item(), steps)
                writer.add_scalar("trainig/p_loss", p_loss.item()*perceptual_weight, steps)
                writer.add_scalar("trainig/adv_loss", adv_loss_.item()*adv_weight, steps)
                writer.add_scalar("trainig/g_loss", g_loss.item(), steps)
                writer.add_scalar("trainig/d_loss", d_loss.item(), steps)
                
                print('epoch {}/{} batch {}/{} rec pet loss {:.5f} p_loss {:.5f} adv_loss {:.5f} g_loss {:.5f} d_loss {:.5f} rec t1 loss {:.5f}'.format(
                    epoch, args.epochs, batch_idx, len(training_dataloader), rec_loss_pet.item(), p_loss.item()*perceptual_weight, 
                    adv_loss_.item()*adv_weight, g_loss.item(), d_loss.item(), rec_loss_t1.item()))
            
                
            
        if epoch % args.eval_every == 0:
            
            pet_decoder.eval()
            discriminator.eval()
            t1_encoder.eval()
            t1_decoder.eval()
            eval_loss = {'rec_loss':[], 'p_loss':[], 'kl_loss':[], 'adv_loss':[], 'g_loss':[], 'd_loss':[], 'rec_t1_loss':[]}
            for batch_idx, batch in enumerate(eval_dataloader):
                
                
                t1_img, pet_img, info, subject, t1_date, pet_date = batch
                t1_img, pet_img = t1_img.to(device), pet_img.to(device)
                info = info.view(t1_img.shape[0], 1, -1).to(device)
                
                with torch.no_grad():
                    latent = t1_encoder(t1_img)
                    z_mu = latent[:, :3]
                    z_sigma = latent[:, 3:]     
                    eps_PET = torch.randn_like(z_sigma)
                    rec_pet = pet_decoder(z_mu+eps_PET*z_sigma, info)
                
            
                rec_pet_loss = l1_loss(rec_pet, pet_img)
                rec_t1_loss = l1_loss(t1_rec, t1_img)
                    
                p_loss = perceptual_loss(rec_pet.float(), pet_img.float())
                
                if adv_weight > 0:
                    with torch.no_grad():
                        logits_fake = discriminator(rec_pet.contiguous().float())[-1]
                    adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                else:
                    adv_loss_ = torch.tensor([0.0]).to(device)

                g_loss = rec_pet_loss + perceptual_weight * p_loss + adv_weight * adv_loss_
                
                
                g_loss = g_loss.mean()
                rec_pet_loss = rec_pet_loss.mean()
                p_loss = p_loss.mean()
                adv_loss_ = adv_loss_.mean()
                rec_t1_loss = rec_t1_loss.mean()
                
                eval_loss["rec_loss"].append(rec_pet_loss.item())
                eval_loss["p_loss"].append(p_loss.item()*perceptual_weight)
                eval_loss["adv_loss"].append(adv_loss_.item()*adv_weight)
                eval_loss["g_loss"].append(g_loss.item())
                eval_loss["rec_t1_loss"].append(rec_t1_loss.item())
                
                
                if adv_weight > 0:
                    with torch.no_grad():
                        logits_fake = discriminator(rec_pet.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(pet_img.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    d_loss = (loss_d_fake + loss_d_real) * 0.5
                    d_loss = d_loss.mean()
                    eval_loss["d_loss"].append(d_loss.item())
            
            if rank == 0:
                steps = epoch*len(training_dataloader) + batch_idx
                writer.add_scalar("eval/rec_loss", np.mean(eval_loss['rec_loss']), steps)
                writer.add_scalar("eval/p_loss", np.mean(eval_loss['p_loss']), steps)
                writer.add_scalar("eval/adv_loss", np.mean(eval_loss['adv_loss']), steps)
                writer.add_scalar("eval/g_loss", np.mean(eval_loss['g_loss']), steps)
                writer.add_scalar("eval/d_loss", np.mean(eval_loss['d_loss']), steps)
                
                
                print('eval epoch {} rec_loss {:.5f} p_loss {:.5f} g_loss {:.5f} d_loss {:.5f}'.format(
                    epoch, 
                    np.mean(eval_loss['rec_loss']), 
                    np.mean(eval_loss['p_loss']), 
                    np.mean(eval_loss['g_loss']), 
                    np.mean(eval_loss['d_loss'])))
            
            
                img_np = pet_img[0].cpu().numpy().squeeze()
                rec_np = rec_pet[0].cpu().numpy().squeeze()
                
                sitk.WriteImage(sitk.GetImageFromArray(img_np), os.path.join(sample_folder, 'epoch={}_ori.nii.gz'.format(epoch)))
                sitk.WriteImage(sitk.GetImageFromArray(rec_np), os.path.join(sample_folder, 'epoch={}_rec.nii.gz'.format(epoch)))
        

        
        if epoch % args.save_every == 0:
            
            state_dict = {'pet_decoder':pet_decoder.state_dict(),
                          'discriminator':discriminator.state_dict(),
                          't1_encoder':t1_encoder.state_dict(),
                          't1_decoder':t1_decoder.state_dict(),
                          'epoch':epoch,
                          'g_optimizer':g_optimizer.state_dict(),
                          'd_optimizer':d_optimizer.state_dict()}
            torch.save(state_dict, os.path.join(os.path.join(ckpt_folder, 'epoch={}.model'.format(epoch))))
            

        
        torch.clear_autocast_cache()
            
    
def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--model_config_path', type=str, 
                        default='./T1toPET/causal_synthesis/configs/training_causal.json')
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[0, 1, 2, 3, 5], nargs='+')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='./T1toPET/causal_synthesis/exp_causal')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--min_and_max', type=str, default='./T1toPET/unet/config/AV45_min_and_max.pkl')
    
    ## pretrained model
    parser.add_argument('--resume', type=str, default=None)
    
    ## dataset
    
    args = parser.parse_args()
    
    args.training_info_csv = f'./T1toPET/unet/config/pair_t1_{args.pet_kind}_training_with_csf.csv'
    args.eval_info_csv = f'./T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'
    
    args.PET_dir = f'./dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'./dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'

    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
    
    