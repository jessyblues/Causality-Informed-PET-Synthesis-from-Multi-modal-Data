
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torch.backends import cudnn
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt
import copy
import torch.multiprocessing as mp
from sklearn import metrics


import torch
from pet_for_classification.dataset import PETdataset, pair_MRI_dataset_only_mci
from torch.utils.data import DataLoader
import pickle
from torch.optim.lr_scheduler import ExponentialLR
import random
from monai_diffusion.generative.networks.nets.atten_unet import DiffusionModelEncoder
from torch.optim import Adam
import csv
import json

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '01111'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'




def test(rank, args):
    

    #print(torch.cuda.device_count())
    #device = torch.device('cuda:0')
    
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)


    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda", args.cuda_ids[rank])
    
    need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.PET_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']    
    
    with open(args.model_config_path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
    if args.use_tabular_info:
        need_values = ['TAU','PTAU','Age','Sex','APOE4','PTEDUCAT'] if args.PET_kind == 'AV1451' else ['ABETA','Age','Sex','APOE4','PTEDUCAT']
        model_dict['atten_encoder_def']['cross_attention_dim'] = len(need_values)
    else:
        need_values = []
    
    if args.use_t1_img and not args.not_use_pet:
        model_dict['atten_encoder_def']["in_channels"] = 2
    else:
        model_dict['atten_encoder_def']["in_channels"] = 1
        
    model = DiffusionModelEncoder(**model_dict['atten_encoder_def']).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.cuda_ids[rank]], find_unused_parameters=False)

    if not args.resume is None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        epoch = ckpt['epoch']
    else:
        print('resume is missing')
        quit()

    with open(args.min_and_max, 'rb') as f:
        min_and_max = pickle.load(f)
        
    pet_name = 'rec.nii.gz' if args.test_fake_pet else None
        
    eval_dataset = pair_MRI_dataset_only_mci(info_csv=args.eval_info_csv, 
                                        pet_dir=args.img_folder, 
                                        t1_dir=args.t1_folder,
                                        need_values=need_values, min_and_max=min_and_max,
                                        use_PET=not args.not_use_pet,
                                        use_T1=args.use_t1_img,
                                        pet_name=pet_name)
    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset,  
                                                    batch_size=args.batch_size, 
                                                    sampler=eval_sampler,
                                                    drop_last=True,
                                                    num_workers=0)
    
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 3], dtype=torch.float)).to(device)


    model.eval()
    epoch_eval_loss = .0
    eval_predict_labels = []
    eval_predict_logits = []
    eval_gt_labels = []
    
    for batch_idx, batch in enumerate(eval_dataloader):
        
        imgs, info, subject, pet_date, t1_date, gts  = batch
        imgs = torch.cat(imgs, dim=1)
        info = info.to(device).unsqueeze(1) if info != [] else None
        gts = gts.to(device)
        
        #print(subject, only_MCI_gts)
        with torch.no_grad():
            predict = model(imgs, torch.zeros((len(subject), ), device=device), info)

        prediction_loss = loss(predict, gts)
        epoch_eval_loss += prediction_loss.item()
        
        eval_predict_labels.append(torch.argmax(predict, dim=1))         
        eval_predict_logits.append(predict)
        eval_gt_labels.append(gts)

    
    eval_predict_labels = torch.cat(eval_predict_labels, dim=0)
    eval_gt_labels = torch.cat(eval_gt_labels, dim=0)
    eval_predict_logits = torch.cat(eval_predict_logits, dim=0)
    
    #pdb.set_trace()
    
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
                                                     eval_predict_logits[:, 1].squeeze().detach().cpu().numpy(), average='weighted')
    
    print('eval f1 {:.5f} auc {:.5f} acc {:.5f} prec {:.5f} recall {:.5f}'.format(
                   eval_metrics['F1'], eval_metrics['auc'], eval_metrics['acc'],  eval_metrics['precision'], eval_metrics['recall']
                ))
    print('eval f1 {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f}'.format(
                   eval_metrics['F1'], eval_metrics['auc'], eval_metrics['acc'],  eval_metrics['precision'], eval_metrics['recall']
                ))
    
    eval_confusion_matrix = np.zeros((2, 2))
    for i in range(eval_gt_labels.shape[0]):
        eval_confusion_matrix[eval_gt_labels[i].item(), eval_predict_labels[i].item()] += 1
            
        
    
    plt.matshow(eval_confusion_matrix, cmap=plt.cm.Greens) 
    plt.colorbar()
    for i in range(eval_confusion_matrix.shape[0]): 
        for j in range(eval_confusion_matrix.shape[1]):
            plt.annotate(eval_confusion_matrix[i,j], xy=(j, i), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if not args.test_fake_pet:
    
        test_folder = os.path.join(args.exp_dir, 'test')
        os.makedirs(test_folder, exist_ok=True)
        plt.savefig(os.path.join(test_folder,
                            'epoch={}_eval_best.jpg'.format(epoch)))
    else:
        test_kind = (args.exp_dir).split('/')[-1]
        plt.savefig(os.path.join(args.fake_image_dir,
                            'epoch={}_eval_only_MCI_test_fake_PET_{}.jpg'.format(epoch, test_kind)))
                            
    plt.close()
    
        
def main():
    
    parser = argparse.ArgumentParser()


    ## config
    parser.add_argument('--training_config', type=str, 
                        default='./diffusion/config/diffusion_training_condition.json')
    
    parser.add_argument('--use_tabular_info', action='store_true', default=False)
    parser.add_argument('--use_t1_img', action='store_true', default=False)
    parser.add_argument('--not_use_pet', action='store_true', default=False)

    #training 
    parser.add_argument('--cuda_ids', type=int, default=[2], nargs='+')
    parser.add_argument('--exp_dir', type=str, help='exp folder', default='./pet_for_classification/exp/pMCI_sMCI')
    parser.add_argument('--resume', type=str, help='resume from checkpoint', default=None)
    parser.add_argument('--test_fake_pet', action='store_true', default=False)
    parser.add_argument('--fake_image_dir', type=str, default='./unet/exp/conditional/AV45/test_output/epoch=22')
    
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    
    parser.add_argument('--save_every_epoch', default=5, type=int)
    parser.add_argument('--eval_every_epoch', default=1, type=int)
    
    ## dataset
    parser.add_argument('--PET_kind', default='AV45', type=str)
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    args.img_folder = f'./dataset/brain_MRI/ADNI/PET/{args.PET_kind}_reg_brain_new1' if not args.test_fake_pet \
        else args.fake_image_dir
    args.t1_folder = './dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'

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
    args.model_config_path = './pet_for_classification/config/training_atten.json' if args.use_tabular_info else \
        './pet_for_classification/config/training_no_atten.json'
    if args.resume is None:
        args.resume = f'{args.exp_dir}/ckpt/best.ckpt'
    mp.spawn(test, args=(args,), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    main()

    

