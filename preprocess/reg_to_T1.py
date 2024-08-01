import ants
import os
import csv
import pdb
import numpy as np

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

def create_T1_downsample(src_img_dir, dst_folder, reg_to_mni_first=False):
    
    with_skull_template = ants.image_read('/home1/yujiali/dataset/brain_MRI/Template/icbm_152_t1.nii')
    
    crop_size = (160, 224, 160)
    set_reso = 1

    template_size = with_skull_template.shape
    
    
    margin = [(template_size[idx] - crop_size[idx])//2 for idx in range(len(template_size))]
    mask = np.zeros(template_size)
    #pdb.set_trace()
    mask[margin[0]+1:-margin[0], margin[1]+1:-margin[1], margin[2]+1:-margin[2]] = np.ones(crop_size)
    ants_mask = ants.from_numpy(mask)
    
    subjects = os.listdir(src_img_dir)
    for subject in sorted(subjects):
        if os.path.exists(os.path.join(dst_folder, subject)):
            continue
        subject_folder = os.path.join(src_img_dir, subject)
        date = sorted(os.listdir(subject_folder))[0]
        baseline_t1_file = os.listdir(os.path.join(subject_folder, date))[0]
        baseline_t1_path = os.path.join(subject_folder, date, baseline_t1_file)
        
        t1_img = ants.image_read(baseline_t1_path)
        if reg_to_mni_first:
            t1_img = reg(t1_img, with_skull_template)['warpedmovout']
        downsampled_img = resample(t1_img, reso=[1.5, 1.5, 1.5])

        dst_folder_ = os.path.join(dst_folder, subject)
        os.makedirs(dst_folder_, exist_ok=True)
        downsampled_img.to_file(os.path.join(dst_folder_, baseline_t1_file))
        
        
        print(subject, 'finished!')

def reg_to_t1(pet_dir, t1_dir, dst_dir):
    
    subjects = sorted(os.listdir(pet_dir))
    missing_t1 = []
    for subject in subjects:
        subject_folder = os.path.join(pet_dir, subject)
        dates = sorted(os.listdir(subject_folder))
        t1_folder = os.path.join(t1_dir, subject)
        if not os.path.exists(t1_folder):
            print(subject, 'not exists t1!')
            missing_t1.append(subject)
            continue
        else:
            t1_file = os.listdir(t1_folder)[0]
            t1_path = os.path.join(t1_folder, t1_file)
            t1_image = ants.image_read(t1_path)
        for date in dates:
            
            date_folder = os.path.join(subject_folder, date)
            pet_file = os.listdir(date_folder)[0]
            
            dst_folder = os.path.join(dst_dir, subject, date)
            if os.path.exists(dst_folder):
                print(subject, date, 'finished!')
                continue
            else:
                os.makedirs(dst_folder, exist_ok=True)
            dst_file = os.path.join(dst_folder, pet_file)
            
            pet_mri = ants.image_read(os.path.join(date_folder, pet_file))
            
            reged_pet_mri = reg(pet_mri, t1_image)['warpedmovout']
            reged_pet_mri.to_file(dst_file)
            
            print(subject, date, 'finished!')
    
    #print(*missing_t1, sep=', ')
    return missing_t1
        

if __name__ == '__main__':
    

    
    #create_T1_downsample(src_img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/T1/new_with_skull/missing_template_for_PET', 
    #                     dst_folder='/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1', 
    #                     reg_to_mni_first=True)
        
    m1 = reg_to_t1(pet_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451', 
              t1_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1', 
              dst_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg') 
    m2 = reg_to_t1(pet_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV45', 
              t1_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1', 
              dst_dir='/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV45_reg') 
    #pdb.set_trace()
    m_ = sorted(list(set(m2 + m1)))
    print(len(m_))
    #print(*m_, sep=', ')
    
    rows = []
    for subject in m_:
        rows.append({'Subject':subject})
        
        
    #with open('./missing.csv', 'w', newline='') as f:
        
    #    fieldnames = rows[-1].keys()
    #    writer = csv.DictWriter(f, fieldnames=fieldnames)
    #    writer.writeheader()
    #    writer.writerows(rows)
    #create_T1_downsample(src_img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/T1/new_with_skull/new_with_skull_obly_bl_mni', 
    #                     dst_folder='/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1')