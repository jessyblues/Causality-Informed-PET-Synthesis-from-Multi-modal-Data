import csv
import os
import random

if __name__ == '__main__':
    
    pet_kind = 'AV45'
    data_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{pet_kind}_reg_brain'
    dst_dir = '/home1/yujiali/T1toPET/diffusion/config'
    lines = []
    training_ratio = 0.8
    subjects = sorted(os.listdir(data_dir))
    random.shuffle(subjects)
    
    training_number = int(training_ratio*len(subjects))
    training_subjects = subjects[:training_number]
    test_subjects = subjects[training_number:]
    
    for subject in sorted(training_subjects):
        subject_folder = os.path.join(data_dir, subject)
        dates = sorted(os.listdir(subject_folder))
        for date in dates:
            date_folder = os.path.join(subject_folder, date)
            pet_file = os.listdir(date_folder)[0]
            id = pet_file.split('.')[0]
            lines.append({'Subject':subject, 'Acq Date':date, 'Image ID':id})
    training_csv_path = os.path.join(dst_dir, f'{pet_kind}_training.csv')
    
    with open(training_csv_path, 'w', newline='') as f:
        fieldnames = list(lines[-1].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    
    for subject in sorted(test_subjects):
        subject_folder = os.path.join(data_dir, subject)
        dates = sorted(os.listdir(subject_folder))
        for date in dates:
            date_folder = os.path.join(subject_folder, date)
            pet_file = os.listdir(date_folder)[0]
            id = pet_file.split('.')[0]
            lines.append({'Subject':subject, 'Acq Date':date, 'Image ID':id})
    test_csv_path = os.path.join(dst_dir, f'{pet_kind}_test.csv')
    with open(test_csv_path, 'w', newline='') as f:
        fieldnames = list(lines[-1].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
        
    

    
    
