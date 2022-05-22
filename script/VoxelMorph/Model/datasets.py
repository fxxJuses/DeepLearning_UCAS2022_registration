import os
import glob
import torch
import sys
from torch.utils.data import Dataset
# from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import nibabel as nib
import numpy as np


class OASIS_Dataset(Dataset):
    def __init__(self, file_dir, transforms=None):
        self.file_dir = file_dir
        self.files_path = glob.glob(os.path.join(
            file_dir, "*/aligned_norm.nii.gz"))
        self.files_path = self.files_path[:294]
        self.seg_path = glob.glob(os.path.join(
            file_dir, "*/aligned_seg35.nii.gz"))[:294]

        self.transforms = transforms

    def load_nii(self, path):
        image = nib.load(path).get_fdata()
        return image

    def Normal(self, x):
        x = (x-x.min())/(x.max() - x.min())
        return x

    def __getitem__(self, index):
        x, x_seg = self.load_nii(self.files_path[index]),  nib.load(
            self.seg_path[index]).get_fdata()

        random_index = random.randint(0, len(self.files_path)-1)
        while random_index == index:
            random_index = random.randint(0, len(self.files_path)-1)

        y = self.load_nii(self.files_path[random_index])

        y_seg = self.load_nii(self.seg_path[random_index])

        # shape of x ,y  ----- (160,192,224)
        #很新颖的表达方式
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)  # (1, 160, 192, 224)  (1, 160, 192, 224)
        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = self.Normal(torch.from_numpy(x)), self.Normal(torch.from_numpy(
            y)), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.files_path)


class OASIS_InferDataset(Dataset):
    def __init__(self, file_dir, transforms=None):
        self.file_dir = file_dir
        self.files_path = glob.glob(os.path.join(
            file_dir, "*/aligned_norm.nii.gz"))[295:]
        self.seg_path = glob.glob(os.path.join(
            file_dir, "*/aligned_seg35.nii.gz"))[295:]
        # print(self.seg_path)

        # self.transforms = transforms

    def load_nii(self, path):
        image = nib.load(path).get_fdata()
        return image

    def Normal(self, x):
        x = (x-x.min())/(x.max() - x.min())
        return x

    def __getitem__(self, index):
        x, x_seg = self.load_nii(self.files_path[index]),  nib.load(
            self.seg_path[index]).get_fdata()

        
        y = self.load_nii(self.files_path[index+1])

       
        y_seg = self.load_nii(self.seg_path[index+1])


        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # [Bsize,channelsHeight,,Width,Depth]
        # x_seg = np.ascontiguousarray(x_seg)
        # y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = self.Normal(torch.from_numpy(x)), self.Normal(torch.from_numpy(
            y)), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.files_path)-1


if __name__ == "__main__":
    dataset = OASIS_Dataset(
        file_dir="/home/fanxx/luoluo/data/all_data")
    dataset.__getitem__(0)

    datainfoset = OASIS_InferDataset(
        file_dir="/home/fanxx/luoluo/data/all_data")
