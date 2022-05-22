from pickletools import optimize
from torch.utils.tensorboard import SummaryWriter
import os
import utils
import glob
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.CosVoxelMorph import CosNetwork
from models.CosVoxelMorph import CONFIGS as CONFIGS
from torch.optim import Adam
from models.losses import ncc_loss, gradient_loss

# os.environ['CUDA_LAUNCH_BLOCKING'] = '9'
# torch.cuda.set_device(9)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():

    file_dir = "/home/fanxx/luoluo/data/all_data"

    batch_size = 1
    
    weights = [1, 0.02]
    save_dir = 'CosMorph_alpha10_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    epoch_start = 0
    max_epoch = 500

    '''
    Initialize model
    '''
    config = CONFIGS['Cos-Morph']
    model = CosNetwork(config).cuda()
    pth_path = "/home/fanxx/fanxx/Registration/CosVoxelMorph/experiments/CosMorph1_mse_1_diffusion_0.02/dsc0.772.pth"
    model.load_state_dict(torch.load(pth_path,map_location=lambda storage, loc: storage).module.state_dict())
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.inputSize, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.inputSize, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([  # trans.RandomFlip(0),
        trans.NumpyType(
            (np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([  # trans.Seg_norm(), #rearrange segmentation label to 1 to 46
        trans.NumpyType((np.float32, np.int16)),
    ])

    # train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    # val_set = datasets.JHUBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    train_set = datasets.OASIS_Dataset(
        file_dir, transforms=train_composed)
    val_set = datasets.OASIS_InferDataset(file_dir, transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=config.batchSize,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True)
    best_dsc = 0

    # Set optimizer 
    opt = Adam(model.parameters(),lr=config.lr)
    sim_loss_fn1 = ncc_loss
    sim_loss_fn2 = ncc_loss
    grad_loss_fn = gradient_loss
    Jacobian_loss = utils.jacobian_determinant_vxm


    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        model.train()
        loss = 0
        for i,data in enumerate(train_loader):
            idx += 1
            loss_reg_all = 0
            data = [t.cuda() for t in data]
            x = data[0].float()
            y = data[1].float()
            # optimizer 
            opt.zero_grad()
            # backward
            df, warp, Fdf, warp_y = model(x, y, "train")
            loss_grad = grad_loss_fn(df)
            loss_sim1 = sim_loss_fn1(y,warp)
            loss_sim2 = sim_loss_fn2(x,warp_y)
            jac_det = utils.Get_Jac(
                df.cpu().permute(0, 2, 3, 4, 1))
            # print(jac_det.size())
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_loss = torch.sum(jac_det <= 0)/(np.prod(tar.shape)*batch_size)

            loss_all = loss_sim1 + 0.5*loss_sim2 + config.alpha * loss_grad #+ jac_loss
            loss += loss_all
            print("i: %d  loss: %f  sim1: %f smi2: %f grad: %f jac:%f" %
                  (i, loss_all.item(), loss_sim1.item(), loss_sim2.item() , loss_grad.item(),jac_loss.item()), flush=True)

            opt.zero_grad()
            loss_all.backward()
            opt.step()

            # x_seg = data[2]
            # y_seg = data[3]

            # def_out = reg_model([x_seg.cuda().float(), df.cuda()])

            # dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            
            # def_out_y = reg_model([y_seg.cuda().float(), Fdf.cuda()])

            # dsc_y = utils.dice_val_VOI(def_out_y.long(), x_seg.long())
            # print("dsc_x : {} , dsc_y:{}".format(dsc,dsc_y))


        writer.add_scalar('Loss/train', loss/i, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss/i))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        jac_all_loss = 0
        model.eval()
        with torch.no_grad():
            for n,data in enumerate(val_loader):
                data = [t.cuda() for t in data]
                x = data[0].float()
                y = data[1].float()
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, config.inputSize)
                flow , _  =model(x,y,'validation')
                jac_det = utils.Get_Jac(
                    flow.permute(0, 2, 3, 4, 1))
            # print(jac_det.size())
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                jac_loss = torch.sum(jac_det <= 0)/(np.prod(tar.shape))
                jac_all_loss += jac_loss
                # visuals = model.get_test_data()
                # flow = visuals['flow_A']
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
            print(jac_all_loss/(n))

        best_dsc = max(eval_dsc.avg, best_dsc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.netG_A.state_dict(),
        #     'best_dsc': best_dsc,
        # }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        torch.save(model, 'experiments/'+save_dir + '/' +
                   'dsc{:.3f}.pth'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        # loss_all.reset()
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    main()
