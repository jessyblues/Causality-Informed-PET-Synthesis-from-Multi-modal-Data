import pandas as pd
import csv
import numpy as np
from datetime import datetime
import pdb

def extract_diagnosis(csv_path='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv'):
    
    subject_diagnosis = {}
    subject_date = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            
            if row['Acq Date'] == '' or row['DX'] == '':
                continue
            
            subject_diagnosis[row['Subject']] = subject_diagnosis[row['Subject']] + [row['DX']] if row['Subject'] in subject_diagnosis.keys() else [row['DX']]
            subject_date[row['Subject']] = subject_date[row['Subject']] + [row['Acq Date']] if row['Subject'] in subject_date.keys() else [row['Acq Date']]
            
    return subject_date, subject_diagnosis

def get_gts(subjects, dates, subject_date, subject_diagnosis):
    
    gts = []
    n = 0 
    
    for subject, date in zip(subjects, dates):
        
        if subject not in subject_date.keys():
            n += 1
            gts.append(0)
            continue
        
        if date in subject_date[subject]:
            diagosis = subject_diagnosis[subject][subject_date[subject].index(date)]
        else:
            all_dates = subject_date[subject]
            diff = [abs((datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(d, '%Y-%m-%d')).days) for d in all_dates]
            pdb.set_trace()
            indx = np.argmin(diff)
            
            diagosis = subject_diagnosis[subject][indx]
            
        if diagosis == 'CN':
            gts.append(0)
        elif diagosis == 'MCI':
            gts.append(1)
        elif diagosis == 'Dementia':
            gts.append(2)
    print(n)
    return gts

def summary(csv_path):
    
    a = pd.read_csv(csv_path)
    #a.set_index(a.columns[0], inplace=True)
    
    values = ['Sex', 'Age']
    
    subject_date, subject_diagnosis = extract_diagnosis()
    subjects, dates = a['Subject'], a['PET_date']
    diagonsis = get_gts(subjects, dates, subject_date, subject_diagnosis)
    
    rows = [a.iloc[row_index].to_dict() for row_index in range(1, len(subjects))]
    r_d = zip(rows, diagonsis)
    
    #pdb.set_trace()
    
    #row_cn = [r for (r, d) in r_d if d == 0]
    r_d = zip(rows, diagonsis)
    row_mci = [r for (r, d) in r_d if d == 1]
    r_d = zip(rows, diagonsis)
    row_ad = [r for (r, d) in r_d if d == 2]
    r_d = zip(rows, diagonsis)
    row_cn = [r for (r, d) in r_d if d == 0]
    
    for (name, row) in zip(['CN', 'MCI', 'AD'], [row_cn, row_mci, row_ad]):
        
        age = [float(r['Age']) for r in row]
        sex = [float(r['Sex']) for r in row]
        subjects = [r['Subject'] for r in row]
        
        print(name, f'Session number {len(age)} Subject number {len(set(subjects))} sex {len(age)-np.sum(sex)}/{np.sum(sex)} age {np.mean(age)}±{np.std(age)}')
        
        
    
summary(csv_path='/home1/yujiali/T1toPET/unet/config/pair_t1_AV1451_all_with_csf.csv')