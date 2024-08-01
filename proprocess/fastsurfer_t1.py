import os
import pdb

downsample_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/PET/downsampled_t1_brain1'
t1_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'

subjects = sorted(os.listdir(downsample_dir))

for subject in subjects:
    
    subject_folder = os.path.join(downsample_dir, subject)
    dates = sorted(os.listdir(subject_folder))
    
    for date in dates:
        
        #pdb.set_trace()
        t1_date_folder = os.path.join(t1_dir, subject, date)
        mni_file = os.listdir(t1_date_folder)[0]
        
        if os.path.exists(f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg/{subject}/{date}'):
            continue
        else:
            os.makedirs(f'/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg/{subject}/{date}')
        
        cmd = f'singularity exec --nv -B:/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI/{subject}:/data \
                                    -B:/home1/yujiali/dataset/brain_MRI/ADNI/T1/fastsurfur_seg/{subject}:/output -B:/home1/yujiali/Tool/freesurfer:/fs_license \
                                    -B:/home1/yujiali/Tool/singularity_images/FastSurfer-dev:/fastsurfer \
                                    /home1/yujiali/Tool/singularity_images/fastsurfer-latest.sif \
                                    /home1/yujiali/Tool/singularity_images/FastSurfer-dev/run_fastsurfer.sh \
                                    --seg_only --t1 /data/{date}/{mni_file} --sid {date} --sd /output'
        os.system(cmd)
        
    