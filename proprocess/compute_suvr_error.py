import csv
import os
import pdb
import numpy as np

gt_csv = '/home1/yujiali/T1toPET/proprocess/AV1451_gt_suvr_iBEAT.csv'
fake_pet_csv_ = '/home1/yujiali/T1toPET/unet/exp/conditional/AV1451/test_output/epoch=456_suvr_iBEAT.csv'
#fake_pet_csv_ = '/home1/yujiali/T1toPET/unet/exp/unconditional/AV1451/test_output/epoch=149_suvr_iBEAT.csv'

gt_suvrs = {}
fake_pet_csv = {}

with open(gt_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        subject=row['Subject']
        date=row['PET_date']
        
        gt_suvrs[(subject, date)] = row['suvr']
        
error_list = []
with open(fake_pet_csv_, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        subject=row['Subject']
        date=row['PET_date']
        
        fake_pet_csv[(subject, date)] = row['suvr']
        
        error_list.append(abs(float(gt_suvrs[(subject, date)]) - float(fake_pet_csv[(subject, date)])))

print(f'{np.mean(error_list):.5f}±{np.std(error_list):.5f}')
        
        