import csv
import datetime
import os
import numpy as np
import glob
import pdb


t1_brain_csv = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/ADNIMERGE.csv'
t1_subject_date = {}
t1_subject_id = {}
exising_nii_files1 = glob.glob('/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain/**/*.nii.gz', recursive=True)
exising_nii_files2 = glob.glob('/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI/**/*.nii.gz', recursive=True)

ids = [exising_nii_file1.split('/')[-1].split('.')[0] for exising_nii_file1 in exising_nii_files1] + [exising_nii_file2.split('/')[-1].split('_')[0] for exising_nii_file2 in exising_nii_files2]
pdb.set_trace()

with open(t1_brain_csv, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            
            subject = row['Subject']
            EXAMDATE = row['EXAMDATE']
            image_id = 'I'+row['IMAGEUID']
            
            if EXAMDATE != '' and image_id != 'I':
                EXAMDATE = datetime.datetime.strptime(row['EXAMDATE'], '%Y/%m/%d')
            else:
                continue
            t1_subject_date[subject] = [EXAMDATE] if subject not in t1_subject_date.keys() else t1_subject_date[subject] + [EXAMDATE]
            t1_subject_id[subject] = [image_id] if subject not in t1_subject_id.keys() else t1_subject_id[subject] + [image_id]
            

pet_dir1 = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV45_reg_brain'
pet_dir2 = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg_brain'
t1_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'

subjects1 = sorted(os.listdir(pet_dir1))
missing_ids1 = []
missing_dates1 = []

for subject in subjects1:
    subject_folder = os.path.join(pet_dir1, subject)
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
                
                if os.path.exists(os.path.join(t1_dir, subject, min_date_str)):
                    continue
                else:
                    missing_ids1.append(t1_subject_id[subject][np.argmin(diff_days)])
                    missing_dates1.append(min_date_str)
                    
subjects2 = sorted(os.listdir(pet_dir2))
missing_ids2 = []
missing_dates2 = []

for subject in subjects2:
    subject_folder = os.path.join(pet_dir2, subject)
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
                
                if os.path.exists(os.path.join(t1_dir, subject, min_date_str)):
                    continue
                else:
                    missing_ids2.append(t1_subject_id[subject][np.argmin(diff_days)])
                    missing_dates2.append(min_date_str)
                    
print(*list(set(missing_ids1)|set(missing_ids2)-set(ids)), sep=', ')
#print(missing_ids)
print(len(list(set(missing_ids1)|set(missing_ids2)-set(ids))))
#print(missing_dates)
    
            
            