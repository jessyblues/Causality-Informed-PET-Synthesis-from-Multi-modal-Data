import dicom2nifti
import os
import pdb
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
import ants

#dicom2nifti.convert_directory(dic_folder, output_folder=dic_folder)

     

def convert_dcm(root):
## convert dcm to nii 
    for subject in os.listdir(root):
        for scan_type in os.listdir(os.path.join(root, subject)):
            nothing_wrong = True
            dates = os.listdir(os.path.join(root, subject, scan_type))
            for date in dates:
                date_ = date.split('_')[0]
                image_folder = os.listdir(os.path.join(root, subject, scan_type, date))[-1]
                image_file = os.listdir(os.path.join(root, subject, scan_type, date, image_folder))[-1]
                try:
                    output_dir = os.path.join(root, subject, date_)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    dicom2nifti.convert_directory(os.path.join(root, subject, scan_type, date, image_folder), output_folder=output_dir)
                
                except Exception as e:
                    nothing_wrong = False
                    print(os.path.join(root, subject, scan_type, date, image_folder, image_file))
                    continue

                output_file_name = os.listdir(output_dir)[0]
                os.rename(os.path.join(output_dir, output_file_name),
                          os.path.join(output_dir, '{}.nii.gz'.format(image_folder)))

            if nothing_wrong:
                shutil.rmtree(os.path.join(root, subject, scan_type))
    

def convert_nii(root):

    for subject in sorted(os.listdir(root)):
        
        folders = os.listdir(os.path.join(root, subject))
    
        for scan_type in os.listdir(os.path.join(root, subject)):
            nothing_wrong = True
            dates = os.listdir(os.path.join(root, subject, scan_type))
            for date in dates:
                date_ = date.split('_')[0]
                image_folder = os.listdir(os.path.join(root, subject, scan_type, date))[-1]
                image_file = os.listdir(os.path.join(root, subject, scan_type, date, image_folder))[0]
                try:
                    img = sitk.ReadImage(os.path.join(root, subject, scan_type, date, image_folder, image_file))
                except Exception as e:
                    nothing_wrong = False
                    print(os.path.join(root, subject, scan_type, date, image_folder, image_file))
                    pdb.set_trace()
                    continue
                if not os.path.exists(os.path.join(root, subject, date_)):
                    os.makedirs(os.path.join(root, subject, date_))
                sitk.WriteImage(img, os.path.join(root, subject, date_, '{}.nii.gz'.format(image_folder)))
            if nothing_wrong:
                shutil.rmtree(os.path.join(root, subject, scan_type))
                print(os.path.join(root, subject))


    
def delete_too_close_scans(root):
    for subject in os.listdir(root):
        dates = os.listdir(os.path.join(root, subject))
        for date in dates:
            images = os.listdir(os.path.join(root, subject, date))
            for idx, image in enumerate(images):
                if idx < len(images)-1:
                    os.remove(os.path.join(root, subject, date, image))
            print(len(os.listdir(os.path.join(root, subject, date))))


    

def count_scans(root):

    info = {}
    counts = []
    for subject in os.listdir(root):
        counts.append(len(os.listdir(os.path.join(root, subject))))
    counts = sorted(counts)
    for counts_ in counts:
        if counts_ not in info.keys():
            info[counts_] = 1
        else:
            info[counts_] += 1
    return info



def resolution_info(root, info_type='resolution'):

    print(root)
    info = {}
    for subject in os.listdir(root):
        dates = os.listdir(os.path.join(root, subject))
        for date in dates:
            if len(os.listdir(os.path.join(root, subject, date))) == 0:
                os.removedirs(os.path.join(root, subject, date))
            else:
                img = os.listdir(os.path.join(root, subject, date))[0]
            ants_img = ants.image_read(os.path.join(root, subject, date, img))
            if info_type == 'spacing':
                info_i = (round(ants_img.spacing[0], 2), round(ants_img.spacing[1], 2), round(ants_img.spacing[2], 2))
            elif info_type == 'shape':
                info_i = ants_img.shape
                #print(info_i)
                if info_i == (192, 192, 160):
                    print(os.path.join(root, subject, date))
                    pdb.set_trace()
            
            if info_i not in info.keys():

                info[info_i] = 1
            else:
                info[info_i] += 1
            
        for k, v in info.items():
            print(k, ':', v, end='  ')
        print('\n') 
    
    return info
