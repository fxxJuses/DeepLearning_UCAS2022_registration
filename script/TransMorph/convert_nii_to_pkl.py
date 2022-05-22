# 遍历文件夹
import os
# nii格式一般都会用到这个包
import nibabel as nib
# 转换成图像
import pickle as pkl

def pkload(fname):
   with open(fname, 'rb') as f:
      return pkl.load(f)

# 主函数
def train_nii_to_image(filepath, pkl_file):
    # 读取nii文件夹
    filenames = os.listdir(filepath)
    nii_list = []
    for filename in filenames:
        if filename.startswith("OASIS"):
            nii_list.append(filename)

    # 开始读取nii文件
    for one_nii in nii_list:

        sub_data = []
        mri_nii_path = os.path.join(filepath, one_nii, 'aligned_norm.nii.gz')
        mri_nii = nib.load(mri_nii_path)  # 读取nii
        mri_data = mri_nii.get_fdata()
        sub_data.append(mri_data)

        seg_nii_path = os.path.join(filepath, one_nii, 'aligned_seg35.nii.gz')
        seg_nii = nib.load(seg_nii_path)
        seg_data = seg_nii.get_fdata()
        sub_data.append(seg_data)

        pkl_path = os.path.join(pkl_file, one_nii)
        pkl_path = pkl_path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pkl.dump(sub_data, f)
            f.close()


def test_nii_to_image(filepath, pkl_file):
    filenames = os.listdir(filepath)
    mri_list = []
    seg_list = []
    for filename in filenames:
        if filename.startswith("img"):
            mri_list.append(filename)
        else:
            seg_list.append(filename)

    for mri in mri_list:
        sub_data = []
        mri_name = mri.replace('.nii.gz', '')
        mri_num = mri_name.replace('img', '')
        for seg in seg_list:
            seg_name = seg.replace('.nii.gz', '')
            seg_num = seg_name.replace('seg', '')
            if mri_num ==seg_num:
                mri_nii_path = os.path.join(filepath, mri)
                mri_nii = nib.load(mri_nii_path)
                mri_data = mri_nii.get_fdata()
                sub_data.append(mri_data)

                seg_nii_path = os.path.join(filepath, seg)
                seg_nii = nib.load(seg_nii_path)
                seg_data = seg_nii.get_fdata()
                sub_data.append(seg_data)

                pkl_path = os.path.join(pkl_file, mri_num)
                pkl_path = pkl_path + '.pkl'
                with open(pkl_path, 'wb') as f:
                    pkl.dump(sub_data, f)
                    f.close()

                tesx = pkload(pkl_path)


if __name__ == '__main__':
    train_filepath = '../DATA_OASIS/Train'
    train_pkl_file = '../DATA_OASIS/Train_pkl'
    # train_nii_to_image(train_filepath, train_pkl_file)

    test_filepath = '../DATA_OASIS/Test_skull_stripped'
    test_pkl_file = '../DATA_OASIS/Test_pkl'
    test_nii_to_image(test_filepath, test_pkl_file)

    a = os.listdir(train_pkl_file)
    b = os.listdir(test_pkl_file)
    m = 2
