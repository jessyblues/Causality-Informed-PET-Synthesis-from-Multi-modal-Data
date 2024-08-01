import os
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import torch
import SimpleITK as sitk
import numpy as np
from monai.transforms import CenterSpatialCrop, SpatialPad
import torch.nn.functional as F

def compute_metrics_pairs(real_image_folder, sys_image_folder, 
                          metrics = ['MAE', 'SSIM', 'PSNR'], device=torch.device('cuda:0'),
                          set_mean = True):

    mae = None
    ssim = None
    psnr = None

    if 'MAE' in metrics:
        mae = []
    if 'SSIM' in metrics:
        ssim = []
    if 'PSNR' in metrics:
        psnr = []

    subjects = os.listdir(sys_image_folder)
    root_folder= sys_image_folder

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    for subject in sorted(subjects):
        subject_folder = os.path.join(root_folder, subject)
        dates = sorted(os.listdir(subject_folder))

        for date in sorted(dates):
            
            date_folder = os.path.join(subject_folder, date)
            sys_file = os.listdir(date_folder)[0]
            
            
            sys_image = sitk.ReadImage(os.path.join(date_folder, sys_file))
            sys_image = sitk.GetArrayFromImage(sys_image)

            if not os.path.exists(os.path.join(real_image_folder, subject, date)):
                continue
            real_file = os.listdir(os.path.join(real_image_folder, subject, date))[0]
            real_image_file = os.path.join(real_image_folder, subject, date, real_file)
            real_image = sitk.ReadImage(real_image_file)    
            real_image = sitk.GetArrayFromImage(real_image)
            
            real_image /= np.max(real_image)
            
            if set_mean:
                sys_image = sys_image/np.mean(sys_image)*np.mean(real_image)
            
            img1 = torch.tensor(sys_image, device=device, dtype=torch.float32).unsqueeze(0)
            img1 = SpatialPad(spatial_size=(192, 224, 192))(img1)
            img1 = CenterSpatialCrop(roi_size=(192, 224, 192))(img1)
            
            img2 = torch.tensor(real_image, device=device, dtype=torch.float32).unsqueeze(0)
            img2 = SpatialPad(spatial_size=(192, 224, 192))(img2)
            img2 = CenterSpatialCrop(roi_size=(192, 224, 192))(img2)

            if 'MAE' in metrics:
                mae.append(F.l1_loss(img1, img2).item())
            if 'SSIM' in metrics:
                if not np.isnan(torch.mean(ms_ssim(img1, img2)).item()):
                    ssim.append(torch.mean(ms_ssim(img1, img2)).item())
            if 'PSNR' in metrics:
                psnr.append(10*torch.log10(1/F.mse_loss(img1, img2))) 
            
            #pdb.set_trace()
    
    return mae, ssim, psnr

