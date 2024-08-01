
import argparse
import torch.nn as nn
import torch.utils.data as Data

import sys
sys.path.append('.')

import argparse
import pdb
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from torch.backends import cudnn
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt
import copy
import torch.multiprocessing as mp
from sklearn import metrics

import sys
sys.path.append('.')

import torch
from monai_diffusion.generative.networks.nets.atten_unet import DiffusionModelEncoder
from pet_for_classification.dataset import pair_MRI_dataset_only_mci
from torch.utils.data import DataLoader
import pickle
from torch.optim.lr_scheduler import ExponentialLR
import random
from torch.optim import Adam
import csv
import json

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12349'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'



def train(rank, args):
    

    #print(torch.cuda.device_count())
    #device = torch.device('cuda:0')
    
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)


    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    ckpt_folder = os.path.join(args.exp_dir, 'ckpt')
    writer_folder = os.path.join(args.exp_dir, 'log')
    visualise_folder = os.path.join(args.exp_dir, 'visual_matrix')
    
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(writer_folder, exist_ok=True)
    os.makedirs(visualise_folder, exist_ok=True)
    
    device = torch.device("cuda", args.cuda_ids[rank])
    
    if rank == 0:
        writer = SummaryWriter(log_dir=writer_folder, flush_secs=10)
    
    if args.use_tabular_info:  
        need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.PET_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']  
    else:
        need_values = []
    
    with open(args.model_config_path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
    
    if args.use_t1_img and not args.not_use_pet:
        model_dict['atten_encoder_def']["in_channels"] = 2
    else:
        model_dict['atten_encoder_def']["in_channels"] = 1
    
    if args.use_tabular_info:
        model_dict['atten_encoder_def']["cross_attention_dim"] = len(need_values)
    model = DiffusionModelEncoder(**model_dict['atten_encoder_def']).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.cuda_ids[rank]], find_unused_parameters=True)

    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        begin_epoch = ckpt['epoch']+1
    else:
        begin_epoch = 0

    optimizer = Adam(
        [
            {"params": model.parameters()}
        ],
        lr = 1e-4
        )
    
    if not args.resume is None:
        optimizer.load_state_dict(ckpt['optimizer'])
    
        del ckpt
    #scheduler = ExponentialLR(optimizer, gamma=0.98)
    
    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)
        
    

    training_dataset = pair_MRI_dataset_only_mci(info_csv=args.training_info_csv, 
                                        pet_dir=args.img_folder, 
                                        t1_dir=args.t1_folder,
                                        need_values=need_values, min_and_max=min_and_max,
                                        use_PET=not args.not_use_pet,
                                        use_T1=args.use_t1_img)
    
    eval_dataset = pair_MRI_dataset_only_mci(info_csv=args.eval_info_csv, 
                                        pet_dir=args.img_folder, 
                                        t1_dir=args.t1_folder,
                                        need_values=need_values, min_and_max=min_and_max,
                                        use_PET=not args.not_use_pet,
                                        use_T1=args.use_t1_img)
            
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset,  
                                                    batch_size=args.batch_size, 
                                                    sampler=train_sampler,
                                                    drop_last=True,
                                                    num_workers=0)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=args.batch_size, 
                                                    sampler=eval_sampler,
                                                    drop_last=True,
                                                    num_workers=0)
    
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 4], dtype=torch.float)).to(device)

    for epoch in range(begin_epoch, args.epochs):

        model.train()
        epoch_loss = .0

        train_predict_labels = []
        train_gt_labels = []
        
        training_dataloader.sampler.set_epoch(epoch)
        eval_dataloader.sampler.set_epoch(epoch)
        
        
        for batch_idx, batch in enumerate(training_dataloader):
            
            imgs, info, subject, pet_date, t1_date, gts  = batch
            #pdb.set_trace()
            #print(imgs[0].shape)
            #pdb.set_trace()
            imgs = [img.to(device) for img in imgs]
            imgs = torch.cat(imgs, dim=1)
            info = info.to(device).unsqueeze(1) if info != [] else None
            gts = gts.to(device)
            
            predict = model(imgs, torch.zeros((imgs.shape[0], ), device=device), info)

            
            prediction_loss = loss(predict, gts)
            epoch_loss += prediction_loss.item()
            
            prediction_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            #print(torch.argmax(predict, dim=1))
            train_predict_labels.append(torch.argmax(predict, dim=1))
            train_gt_labels.append(gts)
            
            if rank == 0:
                steps = epoch*len(training_dataloader) + batch_idx
                writer.add_scalar("training loss",prediction_loss.item(),steps)
                #print('epoch {}/{} batch {}/{} training loss {:.5f}'.format(epoch, args.epochs, batch_idx, len(training_dataloader), prediction_loss.item()))

        train_predict_labels = torch.cat(train_predict_labels, dim=0)
        train_gt_labels = torch.cat(train_gt_labels, dim=0)
        train_metrics = {}
        train_metrics['precision'] = metrics.precision_score(train_gt_labels.squeeze().detach().cpu().numpy(),
                                                             train_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
        train_metrics['recall'] = metrics.recall_score(train_gt_labels.squeeze().detach().cpu().numpy(),
                                                             train_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
        train_metrics['acc'] = metrics.accuracy_score(train_gt_labels.squeeze().detach().cpu().numpy(),
                                                             train_predict_labels.squeeze().detach().cpu().numpy())
        train_metrics['F1'] = metrics.f1_score(train_gt_labels.squeeze().detach().cpu().numpy(),
                                                train_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
        train_metrics['auc'] = metrics.roc_auc_score(train_gt_labels.squeeze().detach().cpu().numpy(),
                                                     train_predict_labels.squeeze().detach().cpu().numpy())
        
        
        train_confusion_matrix = np.zeros((2, 2))
        for i in range(train_gt_labels.shape[0]):
            train_confusion_matrix[train_gt_labels[i].item(), train_predict_labels[i].item()] += 1
            
        

        eval_predict_labels = []
        eval_gt_labels = []
        eval_logits = []
        
        
        if (epoch+1) % args.eval_every_epoch == 0 and rank == 0:
            
            model.eval()
            epoch_eval_loss = .0
            for batch_idx, batch in enumerate(eval_dataloader):
                
                imgs, info, subject, pet_date, t1_date, gts  = batch
                
                imgs = [img.to(device) for img in imgs]
                imgs = torch.cat(imgs, dim=1)
                gts = gts.to(device)

                info = info.to(device).unsqueeze(1) if info != [] else None
                
                with torch.no_grad():
                    predict = model(imgs, torch.zeros((imgs.shape[0], ), device=device), info)

                prediction_loss = loss(predict, gts)
                epoch_eval_loss += prediction_loss.item()
                
                            
                eval_predict_labels.append(torch.argmax(predict, dim=1))
                eval_logits.append(predict)
                eval_gt_labels.append(gts)
                
                
            eval_predict_labels = torch.cat(eval_predict_labels, dim=0)
            eval_gt_labels = torch.cat(eval_gt_labels, dim=0)
            eval_logits = torch.cat(eval_logits, dim=0)
            
            eval_confusion_matrix = np.zeros((2, 2))
            
            for i in range(eval_gt_labels.shape[0]):
                eval_confusion_matrix[eval_gt_labels[i].item(), eval_predict_labels[i].item()] += 1
            
            eval_metrics = {}
            eval_metrics['precision'] = metrics.precision_score(eval_gt_labels.squeeze().detach().cpu().numpy(),
                                                                eval_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
            eval_metrics['recall'] = metrics.recall_score(eval_gt_labels.squeeze().detach().cpu().numpy(),
                                                        eval_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
            eval_metrics['acc'] = metrics.accuracy_score(eval_gt_labels.squeeze().detach().cpu().numpy(),
                                                            eval_predict_labels.squeeze().detach().cpu().numpy())
            eval_metrics['F1'] = metrics.f1_score(eval_gt_labels.squeeze().detach().cpu().numpy(),
                                                    eval_predict_labels.squeeze().detach().cpu().numpy(), average='weighted')
            eval_metrics['auc'] = metrics.roc_auc_score(eval_gt_labels.squeeze().detach().cpu().numpy(),
                                                     eval_logits[:,1].squeeze().detach().cpu().numpy())
            
            
            if rank == 0:
                epoch_eval_loss = epoch_eval_loss/len(eval_dataloader)
                print('epoch {}/{} eval loss {:.5f}'.format(epoch, args.epochs, epoch_eval_loss))
                print('train acc {:.5f} f1 {:.5f} recall {:.5f} prec {:.5f} auc {:.5f}'.format(
                    train_metrics['acc'], train_metrics['F1'], train_metrics['recall'], train_metrics['precision'], train_metrics['auc']
                ))
                print('eval acc {:.5f} f1 {:.5f} recall {:.5f} prec {:.5f} auc {:.5f}'.format(
                    eval_metrics['acc'], eval_metrics['F1'], eval_metrics['recall'], eval_metrics['precision'], eval_metrics['auc']
                ))
                
                writer.add_scalar("eval loss",epoch_eval_loss,       epoch)
                writer.add_scalar("eval acc", eval_metrics['acc'],   epoch)
                writer.add_scalar("eval f1",  eval_metrics['F1'],    epoch)
                writer.add_scalar("eval recall",eval_metrics['recall'], epoch)
                writer.add_scalar("eval prec",eval_metrics['precision'],  epoch)
                writer.add_scalar("eval auc",eval_metrics['auc'],  epoch)

                
                plt.matshow(eval_confusion_matrix, cmap=plt.cm.Greens) 
                plt.colorbar()
                for i in range(eval_confusion_matrix.shape[0]): 
                    for j in range(eval_confusion_matrix.shape[1]):
                        plt.annotate(eval_confusion_matrix[i,j], xy=(j, i), horizontalalignment='center', verticalalignment='center')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')

                plt.savefig(os.path.join(visualise_folder,
                                        'epoch={}_eval.jpg'.format(epoch)))
                plt.close()

                
                plt.matshow(train_confusion_matrix, cmap=plt.cm.Greens) 
                plt.colorbar()
                for i in range(train_confusion_matrix.shape[0]): 
                    for j in range(train_confusion_matrix.shape[1]):
                        plt.annotate(train_confusion_matrix[i,j], xy=(j, i), horizontalalignment='center', verticalalignment='center')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')

                plt.savefig(os.path.join(visualise_folder,
                                        'epoch={}_train.jpg'.format(epoch)))
                plt.close()
        
        
        if (epoch+1) % args.save_every_epoch == 0 and rank == 0:
            
            save_dict  = {
                            'model':model.state_dict(),
                            'epoch':epoch,
                            'optimizer':optimizer.state_dict()
                            }
            torch.save(save_dict, os.path.join(ckpt_folder, 'epoch={}.ckpt'.format(epoch)))
            

        
def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--model_config_path', type=str, 
                        default='./pet_for_classification/training.json')

    
    parser.add_argument('--use_tabular_info', action='store_true', default=False)
    parser.add_argument('--use_t1_img', action='store_true', default=False)
    parser.add_argument('--not_use_pet', action='store_true', default=False)

    #training 
    parser.add_argument('--cuda_ids', type=int, default=[1], nargs='+')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='./pet_for_classification/exp/pMCI_sMCI')
    parser.add_argument('--resume', type=str, help='resume from checkpoint', default=None)
    
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    
    parser.add_argument('--save_every_epoch', default=1, type=int)
    parser.add_argument('--eval_every_epoch', default=1, type=int)
    
    ## dataset
    parser.add_argument('--PET_kind', default='AV45', type=str)
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    

    args.img_folder = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{args.PET_kind}_reg_brain_new1'
    args.t1_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'
    
    args.model_config_path = './pet_for_classification/config/training_atten.json' if args.use_tabular_info else \
        './pet_for_classification/config/training_no_atten.json'
    
    args.eval_info_csv = f'./unet/config/pair_t1_{args.PET_kind}_test_with_csf.csv'
    args.training_info_csv = f'./unet/config/pair_t1_{args.PET_kind}_training_with_csf.csv'

    if not args.use_tabular_info and not args.use_t1_img and not args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/PET_only'
    elif args.use_tabular_info and not args.use_t1_img and not args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/PET_and_tabular_info'
    elif args.use_tabular_info and args.use_t1_img and not args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/PET_and_t1_and_tabular_info'
    elif args.use_tabular_info and args.use_t1_img and args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/t1_and_tabular_info'
    elif args.use_tabular_info and not args.use_t1_img and args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/tabular_info_only'
    elif not args.use_tabular_info and args.use_t1_img and args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/t1_image_only'
    elif not args.use_tabular_info and args.use_t1_img and not args.not_use_pet:
        args.exp_dir = f'{args.exp_dir}/{args.PET_kind}/atten_encoder/PET_and_t1'
        
    args.min_and_max = f'./unet/config/{args.PET_kind}_min_and_max.pkl'
    
    args.cuda_ids = [args.cuda_ids] if type(args.cuda_ids) == int else args.cuda_ids
    args.world_size= len(args.cuda_ids)

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    #train(args=args)

if __name__ == '__main__':
    main()