import os
import csv

import datetime
import os
import numpy as np
import glob
import pdb
import random

pet_kind = 'AV1451'

pet_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{pet_kind}_reg_brain_new1'
t1_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'
t1_subject_date = {}

subjects = sorted(os.listdir(pet_dir))
random.seed(100)
random.shuffle(subjects)
training_subjects = subjects[:int(len(subjects)*0.8)]
test_subjects = subjects[int(len(subjects)*0.8):]

for t1_subject in sorted(os.listdir(t1_dir)):
    subject_folder = os.path.join(t1_dir, t1_subject)
    dates = sorted(os.listdir(subject_folder))
    for date in dates:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        t1_subject_date[t1_subject] = [date] if t1_subject not in t1_subject_date.keys() else t1_subject_date[t1_subject] + [date]
    



pair_pet_dates = []
pair_t1_dates = []
pair_subjects = []
lines = []

training_lines = []
test_lines = []

for subject in sorted(subjects):
    subject_folder = os.path.join(pet_dir, subject)
    dates = os.listdir(subject_folder)
    for date in sorted(dates):
        date_ = datetime.datetime.strptime(date, '%Y-%m-%d')
        if subject not in t1_subject_date.keys():
            continue
        else:
            refer_subjects_dates = t1_subject_date[subject]
            diff_days = [abs((refer_subjects_date-date_).days) for refer_subjects_date in refer_subjects_dates]
            if np.min(diff_days) < 180:
                min_date = refer_subjects_dates[np.argmin(diff_days)]
                min_date_str = datetime.datetime.strftime(min_date, '%Y-%m-%d')

                pair_subjects.append(subject)
                pair_t1_dates.append(min_date_str)
                pair_pet_dates.append(date)
                lines.append({'Subject':subject, 'T1_date':min_date_str, 'PET_date':date})
                
                if subject in training_subjects:
                    training_lines.append({'Subject':subject, 'T1_date':min_date_str, 'PET_date':date})
                else:
                    test_lines.append({'Subject':subject, 'T1_date':min_date_str, 'PET_date':date})
            else:
                continue


with open(f'/home1/yujiali/T1toPET/unet/config/pair_t1_{pet_kind}_training_with_csf.csv', 'w', newline='') as f:

    fieldnames = list(training_lines[-1].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(training_lines)
    
with open(f'/home1/yujiali/T1toPET/unet/config/pair_t1_{pet_kind}_test.csv', 'w', newline='') as f:

    fieldnames = list(test_lines[-1].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_lines)
    
with open(f'/home1/yujiali/T1toPET/unet/config/pair_t1_{pet_kind}_all.csv', 'w', newline='') as f:

    fieldnames = list(training_lines[-1].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(lines)
    