from lib2to3.pgen2.pgen import DFAState
import torch.nn as nn 
import torch 
import torch.nn.functional as F 
from .networks import U_Network , SpatialTransformer
import warnings
from .configs import get_CycleMorph_config
warnings.filterwarnings("ignore")


class CosNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.en_decoder = U_Network(
            dim=3,
            enc_nf = [16,32,32,32],
            dec_nf=[32, 32, 32, 32, 8, 8],
        )
        self.STN = SpatialTransformer(size=cfg.inputSize)
        # self.STN = networks.SpatialTransformer(size=(160, 192, 224))
        
        self.conv1 = self.conv_block(3,3,16,4,2)
        self.conv2 = self.conv_block(3,16,16)
        self.conv3 = self.conv_block(3,19,3)
        


    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer
    
    def forward(self,x,y,flag='train'):
        # input = torch.cat([x,y],dim=1)
        df = self.en_decoder(x,y) #[bs,3,256,256,256]

        warp = self.STN(x,df) #[bs,1,256,256,256]

        if flag == "train":
            Fdf1 = self.conv1(df)
            # print(Fdf1.size())
            Fdf = self.conv2(Fdf1)
            Fdf = nn.Upsample(scale_factor=2, mode='nearest')(Fdf)
            Fdf = torch.cat([df,Fdf],dim=1)
            Fdf = self.conv3(Fdf)
            warp_y = self.STN(warp,Fdf)
            return df,warp,Fdf,warp_y

        return df,warp


CONFIGS = {
    'Cos-Morph': get_CycleMorph_config(),
}


if __name__ == "__main__":
    model = CosNetwork()
    x = torch.rand(size=(2,1,160, 192, 224))
    y = torch.rand(size=(2, 1, 160, 192, 224))
    df,warp,Fdf,warp_y = model(x,y)
    print(df.size())
    print(warp.size())
    print(Fdf.size())
    print(warp_y.size())
