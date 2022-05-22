import glob
import os, utils, torch
import sys, ants
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def main():
    test_dir = '/home/fanxx/luoluo/data/all_data'
    # test_npy_path = '/home/fanxx/fxx/test_npy.npy'
    dict = utils.process_label()
    line = ''
    for i in range(35):
        line = line + ',' + dict[i]
    csv_writter(line, 'SyN')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    # test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_set = datasets.OASIS_InferDataset(test_dir,transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            tar = y
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
            x = ants.from_numpy(x)
            y = ants.from_numpy(y)
            

            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_seg = ants.apply_transforms(fixed=y_ants,
                                            moving=x_ants,
                                            transformlist=reg12['fwdtransforms'],
                                            interpolator='nearestNeighbor',)
                                            #whichtoinvert=[True, False, True, False]

            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)
            def_seg = def_seg.numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val_VOI(def_seg.long(), y_seg.long())
            eval_dsc_def.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            line = utils.dice_val_substruct(def_seg.long(), y_seg.long(), stdy_idx)
            line = line #+ ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))

            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            csv_writter(line, 'SyN')
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(tar.shape), 1)


        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))

        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    main()
