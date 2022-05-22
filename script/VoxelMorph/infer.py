import glob
import os
import utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer
import SimpleITK as sitk

def main():
    test_dir = "/home/fanxx/luoluo/data/all_data"
    weights = [1, 0.02]
    model_folder = 'CycleMorph_mse_{}_diffusion_{}/'.format(
        weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    for i in range(36):
        line = line + ',' + dict[i]
    csv_writter(line, 'experiments/' + model_folder[:-1])

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 8, 8]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec)
    checkpoint_path = "/home/fanxx/luoluo/code/Checkpoint2/dsc_0.7748_190.pth"
    UNet.load_state_dict(torch.load(checkpoint_path))
    UNet = UNet.cuda()
    
    reg_model = utils.register_model((160,192,224), 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
        trans.NumpyType(
            (np.float32, np.int16)),
    ])
    test_set = datasets.OASIS_InferDataset(test_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            UNet.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            # x_in = torch.cat((x, y), dim=1)
            flow = UNet(x, y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(
                def_out.long(), y_seg.long(), stdy_idx)
            line = line  # +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f} Jac"{:.4f}'.format(
                dsc_trans.item(), dsc_raw.item(), np.sum(jac_det <= 0) / np.prod(tar.shape)))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


if __name__ == '__main__':
    '''
    GPU configuration
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    main()
