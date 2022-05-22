import os 
import numpy as np 
import nibabel as nib 
import glob
import random

if __name__ == "__main__":
    file_dir = "/home/ma-user/work/data"
    file_path = glob.glob(os.path.join(file_dir, "*/aligned_norm.nii.gz"))
    seg_path = glob.glob(os.path.join(file_dir, "*/aligned_seg35.nii.gz"))
    rand_index = random.sample(range(len(file_path)),len(file_path))
    
    train_npy = [nib.load(file_path[i]).get_fdata() for i in rand_index[:int(len(file_path)*0.7)]]
    np.save("/home/ma-user/work/train_npy.npy",train_npy)
    print("save train")

    val_npy = [[nib.load(file_path[i]).get_fdata() ,
               nib.load(seg_path[i]).get_fdata() ]for i in rand_index[int(len(file_path)*0.7):int(len(file_path)*0.9)]]
    np.save("/home/ma-user/work/val_npy.npy", val_npy)
    print("save val")
    test_npy = [[nib.load(file_path[i]).get_fdata() , nib.load(seg_path[i]).get_fdata()] for i in rand_index[int(len(file_path)*0.9):]]
    np.save("/home/ma-user/work/test_npy.npy",test_npy)
    print("save test")
