import glob, sys
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
import nibabel as nib
import gc

def main():
    test_dir = '../DATA_OASIS/Test_pkl/'
    if not os.path.exists('results/'):
        os.mkdir('results/')
    save_dir = './results/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    config = CONFIGS_TM['TransMorph-Large']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # file_names = glob.glob(test_dir + '*.pkl')
    file_names = natsorted(os.listdir(test_dir))
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            file_name = file_names[stdy_idx].replace('.pkl', '')
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow_new = np.zeros([160,192,224,3])
            flow_new[:,:,:,0] = flow[0]
            flow_new[:,:,:,1] = flow[1]
            flow_new[:,:,:,2] = flow[2]
            new_image = nib.Nifti1Image(flow_new, np.eye(4))
            flow_file = save_dir + 'flow' + file_name + '.nii.gz'
            nib.save(new_image, flow_file)

            wrap = x_def.cpu().detach().numpy()[0,0]
            new_image = nib.Nifti1Image(wrap, np.eye(4))
            wrap_file = save_dir + 'wrap' + file_name + '.nii.gz'
            nib.save(new_image, wrap_file)

            stdy_idx += 1
        # for data in file_names:
        #     x, y, x_seg, y_seg = utils.pkload(data)
        #     x, y = x[None, None, ...], y[None, None, ...]
        #     x = np.ascontiguousarray(x)
        #     y = np.ascontiguousarray(y)
        #     x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        #     file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
        #     print(file_name)
        #     model.eval()
        #     x_in = torch.cat((x, y),dim=1)
        #     x_def, flow = model(x_in)
        #     flow = flow.cpu().detach().numpy()[0]
        #     flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
        #     print(flow.shape)
        #     np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
        #     stdy_idx += 1

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 7
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()