import os
import csv
import datetime
import numpy as np
import pdb

PET_kind = 'AV1451'

union_csv = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/ADNIMERGE.csv' if PET_kind == 'AV45' else '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv'
csfs = ['ABETA'] if PET_kind == 'AV45' else ['TAU', 'PTAU']
csfs += ['Age','Sex','APOE4', 'PTEDUCAT']
csf_subject_date = {}
csf_subject_values = {}


with open(union_csv, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)
    for row in reader:

        subject = row['Subject']
        date = row['EXAMDATE'] if PET_kind == 'AV45' else row['Acq Date']
        csf_values = [row[k] for k in csfs]
        

        if date != '' and not '' in csf_values:
            if PET_kind == 'AV45':
                EXAMDATE = datetime.datetime.strptime(date, '%Y/%m/%d')
            else:
                EXAMDATE = datetime.datetime.strptime(date, '%Y-%m-%d')
        else:
            continue
        
        for idx, v in enumerate(csf_values):
            if v == 'Female':
                csf_values[idx] = 0
            elif v == 'Male':
                csf_values[idx] = 1
            elif '<' in v:
                csf_values[idx] = 0
            elif '>' in v:
                csf_values[idx] = 2000
            
            if csfs[idx] == 'Age' and PET_kind == 'AV45':
                try:
                    csf_values[idx] = float(v) + float(row['Years_bl'])
                except Exception as e:
                    csf_values[idx] = float(v)
            
            csf_values[idx] = float(csf_values[idx])
                    
        csf_subject_date[subject] = [EXAMDATE] if subject not in csf_subject_date.keys() else csf_subject_date[subject] + [EXAMDATE]
        csf_subject_values[subject] = [csf_values] if subject not in csf_subject_values.keys() else csf_subject_values[subject] + [csf_values]

#pdb.set_trace()
PET_subject_date = {}
old_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{PET_kind}_training.csv'
lines = []
j=0
k_ = 0
with open(old_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        k_ += 1
        subject = row['Subject']
        date = row['PET_date']
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        #PET_subject_date[subject] = [date] if subject in PET_subject_date.keys() else PET_subject_date[subject] + [date]
        if not subject in csf_subject_date.keys():
            j+=1
            #print(subject)
            continue
        diff_days = [abs((refer_subjects_date-date).days) for refer_subjects_date in csf_subject_date[subject]]
        if np.min(diff_days) < 10000:
            min_date = csf_subject_date[subject][np.argmin(diff_days)]
            diff_days = (date-min_date).days
            
            min_csf_value = csf_subject_values[subject][np.argmin(diff_days)]
            min_date_str = datetime.datetime.strftime(min_date, '%Y-%m-%d')
            
            for idx, k in enumerate(csfs):
                row[k] = min_csf_value[idx]
            row['Age'] = row['Age'] + diff_days/365
            lines.append(row)
print(j, k_)
print(len(lines))

lines = sorted(lines, key = lambda e:(e.__getitem__('Subject'),e.__getitem__('Age')))

new_csv = '/home1/yujiali/T1toPET/unet/config/'+old_csv.split('/')[-1].split('.')[0]+'_with_csf.csv'
with open(new_csv, 'w', newline='') as f:
    fieldnames = list(lines[-1])
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(lines)
        
        
    