import ants
import os
import pdb
from datetime import datetime
import numpy as np
import glob

def joint_reg(fixed_image_path, moving_image_path, third_image_path,
              reg_image_path=None, reg_mask_path=None):

    # 读取图像
    fixed_image = ants.image_read(fixed_image_path)
    #moving_image = ants.from_numpy(ants.image_read(moving_image_path).numpy()[::-1, ::-1])
    #moving_image.to_file(moving_image_path)
    moving_image = ants.image_read(moving_image_path)
    
    #third_image = ants.from_numpy(ants.image_read(third_image_path).numpy()[::-1, ::-1])
    #third_image.to_file(third_image_path)
    third_image = ants.image_read(third_image_path)

    # 将固定图像和移动图像配准
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

    # 获取变形场（变换）
    transform = registration['fwdtransforms']

    # 应用变形场于第三个图像
    warped_third_image = ants.apply_transforms(fixed=fixed_image, moving=third_image, transformlist=transform,
                                               interpolator="nearestNeighbor")

    # 保存结果
    if reg_image_path is not None:
        ants.image_write(registration['warpedmovout'], reg_image_path)
    if reg_mask_path is not None:
        ants.image_write(warped_third_image, reg_mask_path)

if __name__ == '__main__':
    
    processed_brain_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_brain'
    processed_seg_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_seg'
    targer_t1_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'
    
    subject_date_brain_file = {}
    for file in sorted(os.listdir(processed_brain_folder)):
        subject = file.split('_MR')[0][5:]
        #if subject == '002_S_4237':
            #pdb.set_trace()
        #try:
        if 'raw' in file:
            date = file.split('raw_')[1].split('_')[0][:8]
        elif 'br' in file:
            date = file.split('br_')[1].split('_')[0][:8]
        else:
            pdb.set_trace()
        year, month, day = date[:4], date[4:6], date[6:]
        subject_date_brain_file[subject] = [file] if not subject in subject_date_brain_file.keys() else subject_date_brain_file[subject] + [file]
        
    
    subject_date_seg_file = {}
    for file in sorted(os.listdir(processed_seg_folder)):
        subject = file.split('_MR')[0][5:]
        if 'raw' in file:
            date = file.split('raw_')[1].split('_')[0][:8]
        elif 'br' in file:
            date = file.split('br_')[1].split('_')[0][:8]
        else:
            pdb.set_trace()
        year, month, day = date[:4], date[4:6], date[6:]
        #subject_date_seg_file[subject] = file
        subject_date_seg_file[subject] = [file] if not subject in subject_date_seg_file.keys() else subject_date_seg_file[subject] + [file]
        
    
    downsampled_subjects = sorted(os.listdir('/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'))
    for subject in downsampled_subjects:
        downsampled_subject_folder = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1/{subject}'
        #dates = sorted(os.listdir(downsampled_subject_folder))
        #for date_ in dates:
        #    downsampled_date_folder = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1/{subject}/{date}'
        #    mri_file = os.listdir(downsampled_date_folder)[0]
        #    idx = mri_file.split('_')[0]
            
        #    target_t1_date_folder = f'{targer_t1_folder}/{subject}/{date}'
        #    target_t1_file = os.listdir(target_t1_date_folder)[0]
        #    target_t1_path = f'{target_t1_date_folder}/{target_t1_file}'
            
            #if subject == '002_S_4237':
            #    pdb.set_trace()
            
        target_t1_subject_folder = f'{targer_t1_folder}/{subject}'
        target_t1_dates = os.listdir(target_t1_subject_folder)
            
        if subject in subject_date_brain_file.keys():
            
            for (processed_brain_file, processed_seg_file) in zip(subject_date_brain_file[subject], subject_date_seg_file[subject]):
            
                
                if 'raw' in processed_brain_file:
                    date = processed_brain_file.split('raw_')[1].split('_')[0][:8]
                elif 'br' in processed_brain_file:
                    date = processed_brain_file.split('br_')[1].split('_')[0][:8]
                else:
                    pdb.set_trace()
                year, month, day = date[:4], date[4:6], date[6:]
                date = f'{year}-{month}-{day}'
                
                diff_days = [abs((datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d')-datetime.strptime(d_, '%Y-%m-%d')).days) for d_ in target_t1_dates]
                min_date = target_t1_dates[np.argmin(diff_days)]
                        
                #if subject == '002_S_4237':
                #    pdb.set_trace()
                if os.path.exists(f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_brain/{subject}/{date}'):
                    continue
                else:
                    os.makedirs(f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_brain/{subject}/{date}')
                    os.makedirs(f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_seg/{subject}/{date}')
                
                joint_reg(fixed_image_path=glob.glob(f'{targer_t1_folder}/{subject}/{min_date}/*.nii.gz')[0], 
                        moving_image_path=f'{processed_brain_folder}/{processed_brain_file}', 
                        third_image_path=f'{processed_seg_folder}/{processed_seg_file}',
                        reg_image_path=f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_brain/{subject}/{date}/reg_brain.nii.gz', 
                        reg_mask_path=f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_seg/{subject}/{date}/reg_seg.nii.gz')
                
                print(subject, date, "finished!")
                #pdb.set_trace()
            
            
            
            
            