import glob
import os

from regex import E
import utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.cycleMorph_model import cycleMorph
from models.cycleMorph_model import CONFIGS as CONFIGS


def main():
    test_dir = "/home/fanxx/luoluo/data/all_data"
    model_idx = -1
    weights = [1, 0.02]
    model_folder = 'CycleMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    for i in range(36):
        line = line + ',' + dict[i]
    csv_writter(line, 'experiments/' + model_folder[:-1])
    
    config = CONFIGS['Cycle-Morph']
    model = cycleMorph()
    model.initialize(config)
    model = model.netG_A
    best_model_path = "/home/fanxx/fanxx/Registration/Registrate/TransMorph_Transformer_for_Medical_Image_Registration/Baseline_registration_models/CycleMorph/experiments/CycleMorph_mse_1_diffusion_0.02/dsc0.693.pth.tar"
    best_model = torch.load(best_model_path)['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.inputSize, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
                                        trans.NumpyType((np.float32, np.int16)),
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
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(tar.shape), x.size(0))

            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(
                dsc_trans.item(), dsc_raw.item()))
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
