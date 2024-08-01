import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb

def get_all_diagnosis(diagnosis_csv):
    
    subject_date = {}
    with open(diagnosis_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['DX'] != '':
                subject_date[row['Subject'], row['Acq Date']] = row['DX']
    
    return subject_date

def search_diagnosis(subject_date, subject, date):
    
    if (subject, date) in subject_date.keys():
        return subject_date[(subject, date)]
    else:
        corresponding_subject_dates = [(s, d) for (s, d) in subject_date.keys() if s == subject]
        if corresponding_subject_dates == [] or date == '':
            return False
        else:
            diff_days = []
            
            target_date = datetime.strptime(date, '%Y-%m-%d')
            for (s, d) in corresponding_subject_dates:
                if d == '':
                    continue
                d_ = datetime.strptime(d, '%Y-%m-%d')
                #dates.append(d_)
                diff_days.append(abs((d_-target_date).days))
            
            if diff_days == []:
                return False
            min_diff_day_idx = np.argmin(diff_days)
            corresponding_subject_date = corresponding_subject_dates[min_diff_day_idx]
        
            return subject_date[corresponding_subject_date]

def get_subject_group(csv_path='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv'):
    
    subject = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject[row['Subject']] = subject[row['Subject']] + [row['DX']] if row['Subject'] in subject.keys() else [row['DX']]
    
    sCN = []
    pCN = []
    pMCI = []
    sMCI = []
    AD = []
    
    for subject, ds in subject.items():
        
        if 'CN' in ds:
            if not 'Dementia' in ds:
                sCN.append(subject)
            elif 'Dementia' in ds:
                pCN.append(subject)

        elif 'MCI' in ds:
            if 'Dementia' not in ds:
                sMCI.append(subject)
            else:
                pMCI.append(subject)
        else:
            AD.append(subject)
            
    return {'sCN':sCN, 'pCN':pCN, 'sMCI':sMCI, 'pMCI':pMCI, 'AD':AD}
        



if __name__ == '__main__':
    
    gt = False
    pet_kind = 'AV45'
    epoch = 22
    
    diagnosis_csv = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv'
    
    if gt:
        suvr_csv = f'/home1/yujiali/T1toPET/proprocess/{pet_kind}_gt_suvr_iBEAT.csv'
    else:
        suvr_csv = f'/home1/yujiali/T1toPET/unet/exp/conditional/{pet_kind}/test_output/epoch={epoch}_suvr_iBEAT.csv' 
    output_dir, _ = os.path.split(suvr_csv)
    
    groups = get_subject_group(csv_path='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv')
    
    diagnosis_suvr = {}
    for group, subjects in groups.items():
        diagnosis_suvr[group] = []
    

    idx = 0
    all_number = 0
    diagnosises = get_all_diagnosis(diagnosis_csv)
    with open(suvr_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject=row['Subject']
            date=row['T1_date']
            
            suvr = float(row['suvr'])
            if suvr > 2:
                idx += 1
                all_number += 1
                continue
            
            for group, subjects in groups.items():
                if subject in subjects:
                    diagnosis_suvr[group].append(float(row['suvr']))
            
            all_number += 1
            #if search_diagnosis(diagnosises, subject, date):
            #    if float(row['suvr']) > 3:
            #        idx += 1
            #        all_number += 1
            #        continue
            #    diagnosis_suvr[search_diagnosis(diagnosises, subject, date)].append(float(row['suvr']))
            #    all_number += 1
    print(idx/all_number)
    plt.figure()
    idx = 0
    #pdb.set_trace()
    for dignosis, suvrs in diagnosis_suvr.items():
        
        #if gt:
        #output_img = f'/home1/yujiali/T1toPET/proprocess/{pet_kind}_gt_suvr.jpg'
        plt.scatter(np.arange(start=idx, stop=idx+len(suvrs)), suvrs)
        idx += len(suvrs)
        
    plt.legend(list(diagnosis_suvr.keys()))
    if gt:
        plt.savefig(f'/home1/yujiali/T1toPET/proprocess/{pet_kind}_gt_suvr_iBEAT.jpg')
    else:
        plt.savefig(f'{output_dir}/epoch={epoch}_suvr_iBEAT.jpg')
            
            
    