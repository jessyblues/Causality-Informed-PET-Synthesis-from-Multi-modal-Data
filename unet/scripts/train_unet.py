import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
import sys
sys.path.append('.')

from unet.utils.dataset import pair_PET_T1dataset
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torch.distributed as dist
import numpy as np
import random
import torch.multiprocessing as mp
from monai_diffusion.generative.networks.nets.atten_unet import AttenUNet
from monai_diffusion.generative.losses.perceptual import PerceptualLoss
from monai_diffusion.generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai_diffusion.generative.losses.adversarial_loss import PatchAdversarialLoss
import SimpleITK as sitk
import pickle



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12349'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def requires_grad(net, flag):
    
    for k,v in net.named_parameters():
        v.requires_grad=flag
    return net



   
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
    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)
        
    model_dict['atten_unet_def']['cross_attention_dim'] = len(need_values)
    atten_unet_dict = model_dict['atten_unet_def']
    unet = AttenUNet(**atten_unet_dict)
    unet.to(device)
    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    
    discriminator = PatchDiscriminator(**model_dict['discriminator']).to(device)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)
    perceptual_loss = PerceptualLoss(**model_dict['perceptual_network']).to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True).to(device)
    
    training_config = model_dict['training']
    base_lr = training_config["base_lr"]
    disc_lr = training_config["disc_lr"]
    perceptual_weight = training_config["perceptual_weight"]
    adv_weight = training_config["adv_weight"]
    
    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        unet.load_state_dict(ckpt['unet'])
        discriminator.load_state_dict(ckpt['discriminator'])
        begin_epoch = ckpt['epoch']+1
    else:
        begin_epoch = 0
        ckpt = None
    
    g_optimizer = Adam(
        [
            {"params": filter(lambda p: p.requires_grad, unet.parameters())}         
        ],
        lr = base_lr
        )
    d_optimizer = Adam(
        [
            {"params": discriminator.parameters()}         
        ],
        lr = disc_lr
        )
    l1_loss = nn.L1Loss()
    
    if ckpt is not None and 'g_optimizer' in ckpt.keys():
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    
    training_dataset = pair_PET_T1dataset(info_csv=args.training_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
    eval_dataset = pair_PET_T1dataset(info_csv=args.eval_info_csv, crop=True, crop_size=(96, 128, 96), 
                                          PET_dir=args.PET_dir, T1_dir=args.T1_dir, min_and_max=min_and_max, need_values=need_values)
    
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
        
        
        train_sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(training_dataloader):
        #if False:
            
            requires_grad(discriminator, False)
            requires_grad(unet, True)
            
            t1_img, pet_img, info, subject, t1_date, pet_date = data
            t1_img, pet_img = t1_img.to(device), pet_img.to(device)
            
            if args.use_condition:
                condition = info.to(device).view(pet_img.shape[0], 1, len(need_values))
            else:
                condition = torch.zeros((pet_img.shape[0], 1, len(need_values)), device=device)
            
            output_pet = unet(t1_img, condition)
            
            rec_loss = l1_loss(output_pet, pet_img)
            #f_rec_loss = fourier_l1_loss(output_pet, pet_img)
            p_loss = perceptual_loss(output_pet.float(), pet_img.float())
            
            if adv_weight > 0:
                logits_fake = discriminator(output_pet.contiguous().float())[-1]
                adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                adv_loss_ = torch.tensor([0.0]).to(device)

            g_loss = rec_loss + p_loss*perceptual_weight + adv_weight * adv_loss_
            
            g_loss = g_loss.mean()
            rec_loss = rec_loss.mean()
            adv_loss_ = adv_loss_.mean()
            p_loss = p_loss.mean()
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            ## training d
            if adv_weight > 0:
                requires_grad(discriminator, True)
                requires_grad(unet, False)
                d_optimizer.zero_grad()                
                with torch.no_grad():
                    output_pet = unet(t1_img, condition)
                
                #print(logits_fake)
                logits_fake = discriminator(output_pet.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_fake.backward()
                logits_real = discriminator(pet_img.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d_real.backward()
                #print(rank)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                

                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()
                
                #d_loss.backward()
                d_optimizer.step()
            else:
                d_loss = torch.tensor([0.0]).to(device)
            
            if rank == 0:
                steps = epoch*len(training_dataloader) + batch_idx
                writer.add_scalar("trainig/rec_loss", rec_loss.item(), steps)
                writer.add_scalar("trainig/p_loss", p_loss.item(), steps)
                writer.add_scalar("trainig/adv_loss", adv_loss_.item()*adv_weight, steps)
                
                writer.add_scalar("trainig/g_loss", g_loss.item(), steps)
                writer.add_scalar("trainig/d_loss", d_loss.item(), steps)
                
                print('epoch {}/{} batch {}/{} rec loss {:.5f} p loss {:.5f} adv_loss {:.5f} g_loss {:.5f} d_loss {:.5f}'.format(
                    epoch, args.epochs, batch_idx, len(training_dataloader), rec_loss.item(), p_loss.item(), 
                    adv_loss_.item()*adv_weight, g_loss.item(), d_loss.item()))
            

        if epoch % args.eval_every == 0:
            
            unet.eval()
            discriminator.eval()
            eval_loss = {'rec_loss':[], 'p_loss':[], 'adv_loss':[], 'g_loss':[], 'd_loss':[]}
            for batch_idx, batch in enumerate(eval_dataloader):
                
                t1_img, pet_img, info, subject, t1_date, pet_date = batch
                t1_img, pet_img = t1_img.to(device), pet_img.to(device)
                
                if args.use_condition:
                    condition = info.to(device).view(pet_img.shape[0], 1, len(need_values))
                else:
                    condition = torch.zeros((pet_img.shape[0], 1, len(need_values)), device=device)
                
                with torch.no_grad():
                    output_pet = unet(t1_img, condition)
                
                rec_loss = l1_loss(output_pet, pet_img)
                p_loss = perceptual_loss(output_pet.float(), pet_img.float())
                
                if adv_weight > 0:
                    with torch.no_grad():
                        logits_fake = discriminator(output_pet.contiguous().float())[-1]
                    adv_loss_ = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                else:
                    adv_loss_ = torch.tensor([0.0]).to(device)

                g_loss = rec_loss + adv_weight * adv_loss_ + p_loss*perceptual_weight
                
                g_loss = g_loss.mean()
                rec_loss = rec_loss.mean()
                adv_loss_ = adv_loss_.mean()
                p_loss = p_loss.mean()
                
                eval_loss["rec_loss"].append(rec_loss.item())
                eval_loss["adv_loss"].append(adv_loss_.item()*adv_weight)
                eval_loss["g_loss"].append(g_loss.item())
                eval_loss["p_loss"].append(p_loss.item())
                
                if adv_weight > 0:
                    with torch.no_grad():
                        logits_fake = discriminator(output_pet.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(pet_img.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    d_loss = (loss_d_fake + loss_d_real) * 0.5
                    d_loss = d_loss.mean()
                else:
                    d_loss = torch.tensor([0.0]).to(device)
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
                rec_np = output_pet[0].cpu().numpy().squeeze()
                t1_np = t1_img[0].cpu().numpy().squeeze()
                
                sitk.WriteImage(sitk.GetImageFromArray(img_np), os.path.join(sample_folder, 'epoch={}_ori.nii.gz'.format(epoch)))
                sitk.WriteImage(sitk.GetImageFromArray(rec_np), os.path.join(sample_folder, 'epoch={}_rec.nii.gz'.format(epoch)))
                sitk.WriteImage(sitk.GetImageFromArray(t1_np), os.path.join(sample_folder, 'epoch={}_t1.nii.gz'.format(epoch)))
            
        
            unet.train()
        
        if epoch % args.save_every == 0 and rank == 0:
            
            state_dict = {'unet':unet.state_dict(),
                          'discriminator':discriminator.state_dict(),
                          'epoch':epoch,
                          'g_optimizer':g_optimizer.state_dict(),
                          'eval_loss':np.mean(eval_loss['rec_loss'])}
            torch.save(state_dict, os.path.join(os.path.join(ckpt_folder, 'epoch={}.model'.format(epoch))))
        
        torch.clear_autocast_cache()
    

def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--model_config_path', type=str, 
                        default='/home1/yujiali/T1toPET/unet/config/training.json')
    parser.add_argument('--use_condition', action='store_true', default=False)
    
    ## training
    parser.add_argument('--cuda_ids', type=int, default=[0, 1, 2, 3, 5, 6], nargs='+')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--pet_kind', type=str, default='AV45')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='/home1/yujiali/T1toPET/unet/exp')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    
    ## resume
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)
    
    #if args.use_condition:
    args.training_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_training_with_csf.csv'
    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test_with_csf.csv'
    #else:
    #    args.training_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_training.csv'
    #    args.eval_info_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{args.pet_kind}_test.csv'
    args.PET_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.pet_kind}_reg_brain_new1'
    args.T1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'
    if args.use_condition:
        args.exp_dir = f'{args.exp_dir}/conditional/{args.pet_kind}'
    else:
        args.exp_dir = f'{args.exp_dir}/unconditional/{args.pet_kind}'
    args.min_and_max = f'/home1/yujiali/T1toPET/unet/config/{args.pet_kind}_min_and_max.pkl'
    
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    
    main()
    
    