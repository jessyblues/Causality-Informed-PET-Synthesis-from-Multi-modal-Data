import ants
import os
import csv
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize, RandSpatialCrop
import torch

def N4(ants_img):
    return ants.n4_bias_field_correction(ants_img)


def reg(scr_img, template_img, tranform='Rigid'):
    #pdb.set_trace()
    return ants.registration(fixed=template_img, moving=scr_img, type_of_transform = tranform)

def resample(ants_img, reso=[1, 1, 1]):
    
    
    new_dim = (int(ants_img.spacing[0]/reso[0]*ants_img.shape[0]), 
                int(ants_img.spacing[1]/reso[1]*ants_img.shape[1]), 
                int(ants_img.spacing[2]/reso[2]*ants_img.shape[2]))

    new_img = ants.resample_image(ants_img, new_dim, True, 0)
    return new_img

def create_T1_downsample(src_img_dir, csv_path, dst_folder):
    
    subject_date = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            
            subject = row['Subject']
            t1_date = row['PET_date']
            
            subject_date.append((subject, t1_date))
            
    for (subject, date) in subject_date:
        if os.path.exists(os.path.join(dst_folder, subject, date)):
            continue
        if not os.path.exists(os.path.join(src_img_dir, subject, date)):
            continue
        src_folder = os.path.join(src_img_dir, subject, date)
        baseline_t1_file = os.listdir(src_folder)[0]
        baseline_t1_path = os.path.join(src_folder, baseline_t1_file)
        t1_img = ants.image_read(baseline_t1_path)
        #downsampled_img = resample(t1_img, reso=[1.5, 1.5, 1.5])
        
        t1_img = CenterSpatialCrop(roi_size=(1, 96, 128, 96))(torch.tensor(t1_img.numpy()).unsqueeze(0).unsqueeze(0)).squeeze()
        downsampled_img = ants.from_numpy(t1_img.numpy())
        

        os.makedirs(os.path.join(dst_folder, subject, date))
        downsampled_img.to_file(os.path.join(dst_folder, subject, date, baseline_t1_file))
        
        
        print(subject, 'finished!')

def reg_to_T1(pet_img_dir, t1_img_dir, csv_path, dst_dir):
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            
            subject = row['Subject']
            t1_date = row['T1_date']
            pet_date = row['PET_date']
            
            dst_folder = os.path.join(dst_dir, subject, pet_date)
            if os.path.exists(dst_folder):
                continue
            else:
                os.makedirs(dst_folder)
            
            t1_folder = os.path.join(t1_img_dir, subject, t1_date)
            t1_file = os.listdir(t1_folder)[0]
            t1_path = os.path.join(t1_folder, t1_file)
            pet_folder = os.path.join(pet_img_dir, subject, pet_date)
            pet_file = os.listdir(pet_folder)[0]
            pet_path = os.path.join(pet_folder, pet_file)
            
            t1_img = ants.image_read(t1_path)
            pet_img = ants.image_read(pet_path)
            
            reged_pet = reg(pet_img, t1_img)['warpedmovout']
            dst_path = os.path.join(dst_folder, pet_file)
            reged_pet.to_file(dst_path)
            print(dst_path, 'finished!')
            

if __name__ == '__main__':
    
    reg_to_T1(pet_img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV45_reg_brain_new', 
              t1_img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1', 
              csv_path='/home1/yujiali/T1toPET/unet/config/pair_t1_AV45_all.csv', 
              dst_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV45_reg_brain_new1')
    
    #create_T1_downsample(src_img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg_brain',
    #                     csv_path='/home1/yujiali/T1toPET/unet/config/pair_t1_AV1451_all.csv',
    #                     dst_folder='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg_brain_new')