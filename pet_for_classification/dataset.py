

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
import pandas as pd


class pair_MRI_dataset_only_mci(data.Dataset):
    
    def __init__(self, info_csv, 
                 crop1=True, crop_size1=(96, 128, 96), resize1=False, resize_size1=None,
                 crop2=True, crop_size2=(96, 128, 96), resize2=False, resize_size2=None,
                 need_values = [],
                 pet_dir='',
                 t1_dir='',
                 min_and_max={},
                 latent_folder=None, 
                 return_MRI=True,
                 use_PET=True,
                 use_T1=True,
                 pet_name=None,
                 converter_csv='/home1/yujiali/T1toPET/pet_for_classification/config/mci_converter.csv'):
        
        super().__init__()

        self.info_csv = info_csv
        self.crop1 = crop1
        self.crop_size1 = crop_size1
        self.resize1 = resize1
        self.resize_size1 = resize_size1
        
        self.crop2 = crop2
        self.crop_size2 = crop_size2
        self.resize2 = resize2
        self.resize_size2 = resize_size2
        
        self.need_values = need_values
        self.lines = []
        self.min_and_max = min_and_max
        self.latent_folder = latent_folder
        self.pet_dir = pet_dir
        self.t1_dir = t1_dir
        self.return_MRI = return_MRI
        
        self.use_PET=use_PET
        self.use_T1=use_T1
        
        df = pd.read_csv(converter_csv)
        df.set_index(df.columns[0], inplace=True)
        data_dict = df.to_dict(orient='index')
        
        
        with open(self.info_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                
                line = {}
                
                subject = row['Subject']
                pet_date = row['PET_date']
                t1_date = row['T1_date']
                
                line['Subject'] = subject
                line['PET_date'] = pet_date
                line['T1_date'] = t1_date
                
                
                if not os.path.exists(os.path.join(self.pet_dir, subject, pet_date)) or not os.path.exists(os.path.join(self.t1_dir, subject, t1_date)):
                    continue
                if subject not in data_dict.keys():
                    continue
                
                else:
                    #print(data_dict[subject]['Converter_date'])
                    try:
                        if data_dict[subject]['MCI_converter']==1 and pet_date >= data_dict[subject]['Converter_date']:
                        #pdb.set_trace()
                            continue
                    except Exception as e:
                        pdb.set_trace()
                
                
                
                if pet_name is None:
                    pet_file = os.listdir(os.path.join(self.pet_dir, subject, pet_date))[0] 
                else:
                    pet_file = pet_name
                line['PET_path'] = os.path.join(self.pet_dir, subject, pet_date, pet_file)
                
                t1_file = os.listdir(os.path.join(self.t1_dir, subject, t1_date))[0] 
                line['T1_path'] = os.path.join(self.t1_dir, subject, t1_date, t1_file)
                line['gt'] = data_dict[subject]['MCI_converter']
                
                
                for k in need_values:
                    line[k] = row[k]
               # #pdb.set_trace()
                self.lines.append(line)
                
    
    def _preprocess_img(self, img, crop, resize, crop_size, resize_size):
        
        img = torch.tensor(img)
        if crop:
            img = SpatialPad(spatial_size=crop_size)(img.unsqueeze(0)).squeeze()
            img = CenterSpatialCrop(roi_size=crop_size)(img.unsqueeze(0)).squeeze()
        if resize:
            img = Resize(spatial_size=resize_size)(img.unsqueeze(0)).squeeze()
            
        img = img/torch.max(img)
        img = img.unsqueeze(0)

        return img
    
    def __getitem__(self, index):
        
        line = self.lines[index]
        subject = line['Subject']
        pet_date = line['PET_date']
        t1_date = line['T1_date']

        pet_path = line['PET_path']
        t1_path = line['T1_path']
        
        imgs = []
        if self.use_PET:
            pet_img = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
            pet_img = self._preprocess_img(pet_img, self.crop1, self.resize1, self.crop_size1, self.crop_size1)
            imgs.append(pet_img)
        if self.use_T1:
            t1_img = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            t1_img = self._preprocess_img(t1_img, self.crop2, self.resize2, self.crop_size2, self.crop_size2)
            imgs.append(t1_img)

        
        infos = []
        for k in self.need_values:
            if line[k] == '':
                print(subject, t1_date, pet_date, 'not complete!')
                quit()
            v = float(line[k])
            v = (v - self.min_and_max[k][0])/(self.min_and_max[k][1]-self.min_and_max[k][0]) if k in self.min_and_max.keys() else v
            
            if k == 'ABETA':
                v = 1 - v

            infos.append(v)
            
        if self.need_values == []:
            return imgs, [], subject, pet_date, t1_date, torch.tensor(line['gt'], dtype=torch.long)

        else:
            return  imgs, torch.tensor(infos, dtype=torch.float), subject, pet_date, t1_date, torch.tensor(line['gt'], dtype=torch.long)


    def __len__(self):
        return len(self.lines)