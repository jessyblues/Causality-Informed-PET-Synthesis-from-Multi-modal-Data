import csv
import os
import pdb

import numpy as np

import torch
from torch.utils import data
import SimpleITK as sitk
import pandas as pd
import torch.nn.functional as F
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize, RandSpatialCrop

class pair_PET_T1dataset(data.Dataset):
    
    def __init__(self, info_csv, 
                 crop=True, crop_size=(96, 128, 96), resize=False, resize_size=None,
                 random_crop=False, random_crop_size=(64, 64, 64),
                 need_values = [],
                 PET_dir='',
                 T1_dir='',
                 min_and_max={},
                 return_MRI=True):
        
        super().__init__()

        self.info_csv = info_csv
        self.crop = crop
        self.crop_size = crop_size
        self.random_crop=random_crop
        self.random_crop_size=random_crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.need_values = need_values
        self.lines = []
        self.min_and_max = min_and_max
        self.PET_dir = PET_dir
        self.T1_dir = T1_dir
        self.return_MRI = return_MRI
        
        
        with open(self.info_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                
                line = {}
                
                subject = row['Subject']
                t1_date = row['T1_date']
                pet_date = row['PET_date']
                
                line['Subject'] = subject
                line['T1_date'] = t1_date
                line['PET_date'] = pet_date
                
                if not os.path.exists(os.path.join(self.T1_dir, subject, t1_date)) or not os.path.exists(os.path.join(self.PET_dir, subject, pet_date)):
                    continue
                t1_image_file = os.listdir(os.path.join(self.T1_dir, subject, t1_date))[0] 
                line['T1_ImagePath'] = os.path.join(self.T1_dir, subject, t1_date, t1_image_file)
                
                pet_image_file = os.listdir(os.path.join(self.PET_dir, subject, pet_date))[0] 
                line['PET_ImagePath'] = os.path.join(self.PET_dir, subject, pet_date, pet_image_file)
                    
                for k in need_values:
                    line[k] = row[k]
                self.lines.append(line)
                
    
    def _preprocess_img(self, img1, img2):
        
        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)
        img = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)], dim=0)
        #print(self.crop_size, self.random_crop_size)
        
        

        if self.crop:
            #print(img.shape)
            img = SpatialPad(spatial_size=(2, )+self.crop_size)(img.unsqueeze(0)).squeeze()
            #print(img.shape)
            img = CenterSpatialCrop(roi_size=(2, )+self.crop_size)(img.unsqueeze(0)).squeeze()
            #print(img.shape)
        if self.random_crop:
            img = RandSpatialCrop(roi_size=(2, )+self.random_crop_size, random_size=False, random_center=True)(img.unsqueeze(0)).squeeze()
            #print(img.shape)
        if self.resize:
            img = Resize(spatial_size=(2, )+self.resize_size)(img.unsqueeze(0)).squeeze()
            #print(img.shape)
        
        #print(img.shape)
        #quit()
        img1 = img[0].unsqueeze(0)
        img2 = img[1].unsqueeze(0)
        
        img1 = img1/torch.max(img1)
        #img1 = img1.unsqueeze(0)
        
        img2 = img2/torch.max(img2)
        #img2 = img2.unsqueeze(0)

        return img1, img2
    
    def __getitem__(self, index):
        
        line = self.lines[index]
        subject = line['Subject']
        t1_date = line['T1_date']
        pet_date = line['PET_date']

        t1_path = line['T1_ImagePath']
        pet_path = line['PET_ImagePath']
        
        if self.return_MRI:
            t1_img = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            pet_img = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
            
            t1_img, pet_img = self._preprocess_img(t1_img, pet_img)
            
            #t1_img = self._preprocess_img(t1_img)
            #pet_img = self._preprocess_img(sitk.GetArrayFromImage(sitk.ReadImage(pet_path)))
        else:
            t1_img = torch.zeros(0)
            pet_img = torch.zeros(0)
        

        
        infos = []
        for k in self.need_values:
            v = float(line[k])
            if k in self.min_and_max.keys():
                v = (v - self.min_and_max[k][0])/(self.min_and_max[k][1]-self.min_and_max[k][0])
            infos.append(v)
            
        if self.need_values == []:
            return t1_img, pet_img, [], subject, t1_date, pet_date
        else:
            return t1_img, pet_img, torch.tensor(infos, dtype=torch.float), subject, t1_date, pet_date

    def __len__(self):
        return len(self.lines)
    

class pair_PETlatent_T1_dataset(data.Dataset):
    
    def __init__(self, info_csv, 
                 crop=True, crop_size=(1, 96, 128, 96), resize=False, resize_size=None,
                 random_crop=False, random_crop_size=(1, 64, 64, 64),
                 need_values = [],
                 PET_latent_dir='',
                 T1_dir='',
                 min_and_max={},
                 return_MRI=True):
        
        super().__init__()

        self.info_csv = info_csv
        self.crop = crop
        self.crop_size = crop_size
        self.random_crop=random_crop
        self.random_crop_size=random_crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.need_values = need_values
        self.lines = []
        self.min_and_max = min_and_max
        self.PET_latent_dir = PET_latent_dir
        self.T1_dir = T1_dir
        self.return_MRI = return_MRI
        
        
        with open(self.info_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                
                line = {}
                
                subject = row['Subject']
                t1_date = row['T1_date']
                pet_date = row['PET_date']
                
                line['Subject'] = subject
                line['T1_date'] = t1_date
                line['PET_date'] = pet_date
                
                if not os.path.exists(os.path.join(self.T1_dir, subject, t1_date)) or not os.path.exists(os.path.join(self.PET_latent_dir, subject, pet_date)):
                    continue
                t1_image_file = os.listdir(os.path.join(self.T1_dir, subject, t1_date))[0] 
                line['T1_ImagePath'] = os.path.join(self.T1_dir, subject, t1_date, t1_image_file)
                
                PET_latent_file = os.listdir(os.path.join(self.PET_latent_dir, subject, pet_date))[0] 
                line['PET_LatentPath'] = os.path.join(self.PET_latent_dir, subject, pet_date, PET_latent_file)
                    
                for k in need_values:
                    line[k] = float(row[k])
                    if k in min_and_max.keys():
                        line[k] = (line[k] - min_and_max[k][0])/(min_and_max[k][1]-min_and_max[k][0])
                self.lines.append(line)
                
    
    def _preprocess_img(self, img):
        
        img = torch.tensor(img)
        if self.crop:
            img = SpatialPad(spatial_size=self.crop_size)(img.unsqueeze(0)).squeeze()
            img = CenterSpatialCrop(roi_size=self.crop_size)(img.unsqueeze(0)).squeeze()
        if self.random_crop:
            img = RandSpatialCrop(roi_size=self.random_crop_size, random_size=False, random_center=True)(img.unsqueeze(0)).squeeze()
        if self.resize:
            img = Resize(spatial_size=self.resize_size)(img.unsqueeze(0)).squeeze()
            
        img = img/torch.max(img)
        img = img.unsqueeze(0)

        return img
    
    def __getitem__(self, index):
        
        line = self.lines[index]
        subject = line['Subject']
        t1_date = line['T1_date']
        pet_date = line['PET_date']

        t1_path = line['T1_ImagePath']
        pet_latent_path = line['PET_LatentPath']
        pet_latent = np.load(pet_latent_path)
        pet_latent = torch.tensor(pet_latent, dtype=torch.float)
        
        if self.return_MRI:
            t1_img = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            t1_img = self._preprocess_img(t1_img)
        else:
            t1_img = torch.zeros(0)

        
        infos = []
        for k in self.need_values:
            infos.append(line[k])
            
        if self.need_values == []:
            return t1_img, pet_latent, [], subject, t1_date, pet_date
        else:
            return t1_img, pet_latent, torch.tensor(infos, dtype=torch.float), subject, t1_date, pet_date

    def __len__(self):
        return len(self.lines)