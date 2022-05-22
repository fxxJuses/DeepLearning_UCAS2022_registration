import glob
import os
import utils
import torch
import sys
import ants
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib
import SimpleITK as sitk


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def save_image(img, ref_img, name):
    # print(img[0, 0, ...].size()) # torch.Size([160, 192, 224, 3])
    img = sitk.GetImageFromArray(img)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(
        "/home/fanxx/fanxx/Registration/CosVoxelMorph/output", name))



if __name__ == "__main__":
    moving_path = "/home/fanxx/luoluo/data/test/img0438.nii.gz"
    fixed_path = "/home/fanxx/luoluo/data/test/img0439.nii.gz"
    f_img = sitk.ReadImage(fixed_path)
    moving_image = nib.load(moving_path).get_fdata()
    fixed_image = nib.load(fixed_path).get_fdata()
    print("ok")
    moving_image = moving_image
    fixed_image = fixed_image
    x = ants.from_numpy(moving_image)
    y = ants.from_numpy(fixed_image)
    reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(
        160, 80, 40), syn_metric='meansquares')
    flow = np.array(
        nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
    print(flow.shape)  # (160, 192, 224, 1, 3)
    flow = flow[:, :, :, 0, :]
    save_image(flow, ref_img = f_img , name = "Syn_flow.nii.gz")


    

    out = sitk.GetImageFromArray(reg12)
    sitk.WriteImage(out,'/home/fanxx/fanxx/Registration/CosVoxelMorph/output/Syn_reg.nii.gz')

    def_seg = ants.apply_transforms(fixed=y_ants,
                                    moving=x_ants,
                                    transformlist=reg12['fwdtransforms'],
                                    interpolator='nearestNeighbor',)

    
