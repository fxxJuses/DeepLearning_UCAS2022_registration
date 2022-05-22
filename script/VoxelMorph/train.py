# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np  
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer
from Model.datasets import  OASIS_Dataset
from Model.datasets import  OASIS_InferDataset
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    writer = SummaryWriter(os.path.join(args.model_dir, "Summary"))
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像中 vol_size
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    
    train_datasets = OASIS_Dataset(args.train_dir)
    validation_datasets =  OASIS_InferDataset(args.train_dir)
    train_loader = Data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    validation_loader = Data.DataLoader(validation_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # input_fixed, DS, _, _ 
    
    # # [B, C, D, W, H]
    # input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    # input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec)
    # UNet.load_state_dict(torch.load(args.checkpoint_path))
    UNet = UNet.cuda()

    #.to(device)
    STN = SpatialTransformer((160, 192, 224)).to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data
    # train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))[1:]
    #功能：返回一个某一种文件夹下面的某一类型文件路径列表
    # DS = Dataset(files=train_files)
    # print("Number of training images: ", len(DS))
    # DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for i in range(1, args.n_iter + 1):
        # Generate the moving images and convert them to tensors.
        # print('Training Starts')
        # '''
        # Training
        # '''
        loss_train_all = 0
        for j,data in enumerate(train_loader):
            # data是以一个元组表示的，元组里面包括四个元素
            input_fixed = data[0]
            input_moving = data[1]
        
            # [B, C, D, W, H]
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            # print("input_moving",input_moving.shape)
        # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            # print("flow_m2f",flow_m2f.shape)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss
            loss_train_all += loss.item()
            print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        writer.add_scalar('loss/train', loss_train_all/j, i)
        
        '''
        
        validation……………………
        
        '''
        
        STN_label = SpatialTransformer((160, 192, 224), mode="nearest").to(device)
        UNet.eval()
        STN.eval()
        STN_label.eval()
        DSC = []
        with torch.no_grad():

            for testdata in validation_loader:
                moving_val = testdata[0].to(device).float()
                fixed_val = testdata[1].to(device).float()
                moving_seg = testdata[2].to(device).float()
                fixed_seg = testdata[3]

                pred_flow = UNet(moving_val,fixed_val)
                pred_img = STN(moving_val, pred_flow)
                pred_label = STN_label(moving_seg,pred_flow)

                # 计算DSC
                dice = compute_label_dice(fixed_seg, pred_label[0, 0, ...].cpu().detach().numpy())
                print("dice: ", dice)
                DSC.append(dice)
        writer.add_scalar('DSC/validation', np.mean(DSC), i)
        print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))    
        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, 'dsc_%.4f_%d.pth' % (np.mean(DSC),i))
            torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = str(i) + "_m.nii.gz"
            m2f_name = str(i) + "_m2f.nii.gz"
            save_image(input_moving, f_img, m_name)
            save_image(m2f, f_img, m2f_name)
            print("warped images have saved.")


    writer.close()

    f.close()



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
