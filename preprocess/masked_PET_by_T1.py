import os
import SimpleITK as sitk
import shutil

if __name__ == '__main__':
    
    pet_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg'
    t1_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain'
    dst_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/AV1451_reg_brain'
    subjects = sorted(os.listdir(pet_dir))
    
    for subject in subjects:
        pet_subject_folder = os.path.join(pet_dir, subject)
        t1_subject_folder = os.path.join(t1_dir, subject)
        
        if not os.path.exists(t1_subject_folder):
            #shutil.copytree(pet_subject_folder, '/home1/yujiali/dataset/brain_MRI/ADNI/PET/no_t1_AV45')
            #shutil.rmtree(pet_subject_folder)
            continue
        
        t1_bl_date = os.listdir(t1_subject_folder)[0]
        t1_bl_folder = os.path.join(t1_subject_folder, t1_bl_date)
        t1_file = os.listdir(t1_bl_folder)[0]
        
        t1_path = os.path.join(t1_bl_folder, t1_file)
        t1_template = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
        
        t1_mask = t1_template>0
        
        
        pet_dates = sorted(os.listdir(pet_subject_folder))
        for pet_date in pet_dates:
            pet_date_folder = os.path.join(pet_subject_folder, pet_date)
            pet_file = os.listdir(pet_date_folder)[0]
            
            pet_path = os.path.join(pet_date_folder, pet_file)
            dst_folder_ = os.path.join(dst_folder, subject, pet_date)
            if os.path.exists(os.path.join(dst_folder_, pet_file)):
                print(dst_folder_, 'finished!')
                continue
            
            
            pet = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))*t1_mask
            

            os.makedirs(dst_folder_, exist_ok=True)
            
            sitk.WriteImage(sitk.GetImageFromArray(pet), os.path.join(dst_folder_, pet_file))
            print(dst_folder_, 'finished!')
            