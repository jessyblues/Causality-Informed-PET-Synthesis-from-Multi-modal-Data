import os
import ants
import csv
from monai.transforms import SpatialPad
#SpatialPad
import torch
import numpy as np
import pdb
import glob
from datetime import datetime

def resize_and_PET_to_seg(pet_path, seg_img):
    
    pet_img = ants.image_read(pet_path)
    
    upsampeld_pet_img = ants.resample_image(pet_img, resample_params=(int(pet_img.shape[0]*1.5),
                                                                      int(pet_img.shape[1]*1.5), 
                                                                      int(pet_img.shape[2]*1.5)),
                                                    use_voxels=True)
    padding_img = torch.tensor(upsampeld_pet_img.numpy()).unsqueeze(0).unsqueeze(0)
    #pdb.set_trace()
    padding_img = SpatialPad((1, seg_img.shape[0], seg_img.shape[1], seg_img.shape[2]))(padding_img)
    
    padding_img = padding_img.squeeze().numpy()
    #padding_img = ants.from_numpy(np.swapaxes(padding_img, 1, 2)[:, ::-1, ::-1])
    #padding_img = ants.registration(seg_img, padding_img, type_of_transform='Rigid')['warpedmovout']
    #pdb.set_trace()
    
    return padding_img
    
def compute_suvr(pet_path, seg_img_path, roi_labels, ref_labels, print_masked_pet=False):
    
    #pet_img = ants.image_read(pet_path)
    seg_img = ants.image_read(seg_img_path)
    seg_img_np = seg_img.numpy()
    padded_pet_np = resize_and_PET_to_seg(pet_path, seg_img)
    #padded_pet_np = padded_pet.numpy()
    
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


    PET_kind = 'AV1451'
    #epoch = 31
    gt = False
    
    
    iBEAT_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/processed_reg_seg'
    pet_dir = f'/home1/yujiali/dataset/brain_MRI/ADNI/PET/{PET_kind}_reg_brain_new1' if gt else \
        '/home1/yujiali/T1toPET/unet/exp/conditional/AV1451/test_output/epoch=456'
    #pet_dir = '/home1/yujiali/T1toPET/unet/exp/conditional/AV45/test_output/epoch=56'
    
    
    if not gt:
        epoch = int(pet_dir.split('=')[-1])
        
    pet_t1_pair_csv = f'/home1/yujiali/T1toPET/unet/config/pair_t1_{PET_kind}_test_with_csf.csv'
    suvrs = []
    diagnosis = []
    
    subject_d = get_diagnosis()
    
    diagnosis_suvr = {'Dementia':[],
                      'MCI':[],
                      'CN':[]}
    
    lines = []
    target_folder, _ = os.path.split(pet_dir)
    target_csv = f'{target_folder}/epoch={epoch}_suvr_iBEAT.csv' if not gt else \
                f'/home1/yujiali/T1toPET/proprocess/{PET_kind}_gt_suvr_iBEAT.csv'
                
    roi_labels = np.arange(36, 46).tolist()+np.arange(48, 114).tolist()+np.arange(116, 132).tolist()
    ref_labels = [10, 11]
    
    with open(pet_t1_pair_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            subject = row['Subject']
            pet_date = row['PET_date']
            t1_date = row['T1_date']
            
            if not os.path.exists(f'{iBEAT_dir}/{subject}'):
                print(subject)
                continue
            else:
                dates = os.listdir(f'{iBEAT_dir}/{subject}')
                try:
                    diff_date = [abs((datetime.strptime(t1_date, "%Y-%m-%d") - datetime.strptime(d_, "%Y-%m-%d")).days) for d_ in dates]
                except Exception as e:
                    pdb.set_trace()
                min_idx = np.argmin(diff_date)
                seg_mask = glob.glob(f'{iBEAT_dir}/{subject}/{dates[min_idx]}/*.nii.gz')[0]
            
            pet_folder = f'{pet_dir}/{subject}/{pet_date}'
            if gt:
                pet_file = os.listdir(pet_folder)[0]
            else:
                pet_file = 'rec.nii.gz'
            
            #pdb.set_trace()
            
            
            suvr = compute_suvr(pet_path=f'{pet_folder}/{pet_file}',
                                seg_img_path=seg_mask,
                                roi_labels=roi_labels, #左大脑皮层, 右大脑皮层
                                ref_labels=ref_labels,
                                print_masked_pet=False) #左小脑皮层, 右小脑皮层
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
        fieldnames = list(lines[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    
    
    
        
        
        
        
        