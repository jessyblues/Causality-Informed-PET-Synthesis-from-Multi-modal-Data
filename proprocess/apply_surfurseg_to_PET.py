import os
import ants
import csv
from monai.transforms import SpatialPad
#SpatialPad
import torch
import numpy as np
import pdb

def resize_and_pad_PET_to_seg(pet_path, seg_img):
    
    pet_img = ants.image_read(pet_path)
    
    upsampeld_pet_img = ants.resample_image(pet_img, resample_params=(int(pet_img.shape[0]*1.5),
                                                                      int(pet_img.shape[1]*1.5), 
                                                                      int(pet_img.shape[2]*1.5)),
                                                    use_voxels=True)
    padding_img = torch.tensor(upsampeld_pet_img.numpy()).unsqueeze(0).unsqueeze(0)
    padding_img = SpatialPad((1, 256, 256, 256))(padding_img)
    
    padding_img = padding_img.squeeze().numpy()
    padding_img = ants.from_numpy(np.swapaxes(padding_img, 1, 2)[:, ::-1, ::-1])
    #padding_img = ants.registration(seg_img, padding_img, type_of_transform='Rigid')['warpedmovout']
    
    
    return padding_img
    
def compute_suvr(pet_path, seg_img_path, roi_labels, ref_labels, print_masked_pet=False):
    
    #pet_img = ants.image_read(pet_path)
    seg_img = ants.image_read(seg_img_path)
    seg_img_np = seg_img.numpy()
    padded_pet = resize_and_pad_PET_to_seg(pet_path, seg_img)
    padded_pet_np = padded_pet.numpy()
    
    roi_mask = np.zeros_like(padded_pet_np)
    ref_mask = np.zeros_like(padded_pet_np)
    for roi_label in roi_labels:
        roi_mask = roi_mask + (seg_img_np == roi_label)
        #print(np.sum(roi_mask))
    roi_takein = np.sum(roi_mask*padded_pet_np)/np.sum(roi_mask)
    #pdb.set_trace()
    
    seg_img_folder, seg_file = os.path.split(seg_img_path)
    if print_masked_pet:
        roi_img = ants.from_numpy(roi_mask*padded_pet_np)
        roi_img.to_file(f'{seg_img_folder}/roi_pet.nii.gz')
        ants.from_numpy(padded_pet_np).to_file(f'{seg_img_folder}/pet.nii.gz')
        ants.from_numpy(seg_img_np).to_file(f'{seg_img_folder}/pet_seg.nii.gz')
    
    
    for ref_label in ref_labels:
        ref_mask = ref_mask + (seg_img_np == ref_label)
        #print(np.sum(ref_label))
    ref_takein = np.sum(ref_mask*padded_pet_np)/np.sum(ref_mask)
    if print_masked_pet:
        roi_img = ants.from_numpy(ref_mask*padded_pet_np)
        roi_img.to_file(f'{seg_img_folder}/ref_pet.nii.gz')
        
    
    #pdb.set_trace()
    
    return roi_takein/ref_takein
        
def get_diagnosis(csv_path='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/union.csv'):
    
    subject_date = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_date[row['Subject']] = row['DX']
    
    return subject_date


if __name__ == '__main__':


    PET_kind = 'AV45'
    epoch = 22
    gt = False
    
    
    fastsurfur_seg_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg'
    pet_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{PET_kind}_reg_brain_new1' if gt else \
              f'/home1/yujiali/T1toPET/unet/exp/conditional/{PET_kind}/test_output/epoch={epoch}'
    pet_t1_pair_csv = f'/home1/yujiali/T1toPET/controlnet/config/pair_t1_{PET_kind}_test_with_csf.csv'
    suvrs = []
    diagnosis = []
    
    subject_d = get_diagnosis()
    
    diagnosis_suvr = {'Dementia':[],
                      'MCI':[],
                      'CN':[]}
    
    lines = []
    target_csv = f'/home1/yujiali/T1toPET/unet/exp/conditional/{PET_kind}/test_output/epoch={epoch}_suvr.csv' if not gt else \
                f'/home1/yujiali/T1toPET/proprocess/{PET_kind}_gt_suvr.csv'
    
    with open(pet_t1_pair_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            subject = row['Subject']
            pet_date = row['PET_date']
            t1_date = row['T1_date']
            
            seg_mask = f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg/{subject}/{t1_date}/mri/aseg.auto_noCCseg.mgz'
            seg_folder = f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg/{subject}/{t1_date}/mri'
            
            pet_folder = f'{pet_dir}/{subject}/{pet_date}'
            if gt:
                pet_file = os.listdir(pet_folder)[0]
            else:
                pet_file = 'rec.nii.gz'
            
            #pdb.set_trace()
            cmd = f'mri_convert {seg_mask} {seg_folder}/seg.nii.gz'
            if not os.path.exists(f'{seg_folder}/seg.nii.gz'):
                os.system(cmd)
            
            suvr = compute_suvr(pet_path=f'{pet_folder}/{pet_file}',
                                seg_img_path=f'{seg_folder}/seg.nii.gz',
                                roi_labels=[3, 42], #左大脑皮层, 右大脑皮层
                                ref_labels=[7, 8, 46, 47],
                                print_masked_pet=True) #左小脑皮层, 右小脑皮层
            #print(suvr)
            if subject in subject_d.keys():
                suvrs.append(suvr)
                diagnosis.append(subject_d[subject])
                #pdb.set_trace()
                diagnosis_suvr[subject_d[subject]].append(suvr)
            
            row['suvr'] = suvr
            lines.append(row)
    #pdb.set_trace()
    
    
    with open(target_csv, 'w', newline='') as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    
    
    
        
        
        
        
        