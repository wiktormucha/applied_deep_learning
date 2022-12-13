'''
This file contains network models that are ready to be inported to other pytohn files 
'''
import torch
import torch.nn as nn
from utils.prep_utils import MODEL_NEURONS
from heads import SimpleHead_FPN, SimpleHead5, PredictionHead, SimpleHead_trans
from backbones import BackboneModel_FPN, BackboneModel, ConvBlock

class CustomHeatmapsModel_FPN(nn.Module):
    '''
    Model with feature stacking form different level during extracting
    '''
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel_FPN()
        self.head = SimpleHead_FPN(1984,21)

    def forward(self,x):

        bck = self.backbone(x)
        out = self.head(bck)
       
        return out
    
class CustomHeatmapsModel(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel()
        self.head = SimpleHead5(1280, 21)

    def forward(self,x):

        bck = self.backbone(x)
        out = self.head(bck)
        
        return out

class CustomHeatmapsModel_trans(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel()
        self.att = torch.nn.MultiheadAttention(embed_dim = 1280, num_heads=2, batch_first = True)
        self.head = SimpleHead_trans(1280, 21)

    def forward(self,x):

        # print('Input: ', x.shape)
        bck = self.backbone(x)
      
        re = bck.reshape(x.shape[0],1280)
        att, _ = self.att(key = re, value = re, query = re)
        att = att.reshape(x.shape[0],1280,1,1)
        out = self.head(bck + att)
        
        return out
        

class BaselineFutureStaking(nn.Module):
    """
    Implementation of UNet, slightly modified:
    - less downsampling blocks
    - less neurons in the layers
    - Batch Normalization added
    
    Link to paper on original UNet:
    https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_down4 = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)
        self.conv_down5 = ConvBlock(MODEL_NEURONS * 8, MODEL_NEURONS * 16)
        # self.conv_bottleneck = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsamle4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.upsamle8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.upsamle16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        # self.conv_up1 = ConvBlock(
        #     MODEL_NEURONS * 8 + MODEL_NEURONS * 4, MODEL_NEURONS * 4
        # )
        # self.conv_up2 = ConvBlock(
        #     MODEL_NEURONS * 4 + MODEL_NEURONS * 2, MODEL_NEURONS * 2
        # )
        # self.conv_up3 = ConvBlock(MODEL_NEURONS * 2 + MODEL_NEURONS, MODEL_NEURONS)

        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(496, out_channel, kernel_size=3, padding=1, bias=False),
        #     nn.Sigmoid(),
        # )

        self.head = PredictionHead(input_dim = 496, output_dim = out_channel)

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_d4 = self.conv_down4(self.maxpool(conv_d3))
        conv_d5 = self.conv_down5(self.maxpool(conv_d4))
    
        conv_d2 = self.upsamle(conv_d2)
        conv_d3 = self.upsamle4(conv_d3)
        conv_d4 = self.upsamle8(conv_d4)
        conv_d5 = self.upsamle16(conv_d5)
        # print('conv1: ',conv_d1.shape)
        # print('conv2: ',conv_d2.shape)
        # print('conv3: ',conv_d3.shape)
        # print('conv4: ',conv_d4.shape)
        # conv_u1 = self.conv_up1(torch.cat([self.upsamle(conv_b), conv_d3], dim=1))
        # conv_u2 = self.conv_up2(torch.cat([self.upsamle(conv_u1), conv_d2], dim=1))
        # conv_u3 = self.conv_up3(torch.cat([self.upsamle(conv_u2), conv_d1], dim=1))
        concat = torch.cat([conv_d1,conv_d2,conv_d3,conv_d4,conv_d5], dim=1)
        # out = self.conv_out(concat)
        out = self.head(concat)

        return out

class BaselineUNET(nn.Module):
    """
    Implementation of UNet, slightly modified:
    - less downsampling blocks
    - less neurons in the layers
    - Batch Normalization added
    
    Link to paper on original UNet:
    https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_bottleneck = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv_up1 = ConvBlock(
            MODEL_NEURONS * 8 + MODEL_NEURONS * 4, MODEL_NEURONS * 4
        )
        self.conv_up2 = ConvBlock(
            MODEL_NEURONS * 4 + MODEL_NEURONS * 2, MODEL_NEURONS * 2
        )
        self.conv_up3 = ConvBlock(MODEL_NEURONS * 2 + MODEL_NEURONS, MODEL_NEURONS)

        self.conv_out = nn.Sequential(
            nn.Conv2d(MODEL_NEURONS, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_b = self.conv_bottleneck(self.maxpool(conv_d3))

        conv_u1 = self.conv_up1(torch.cat([self.upsamle(conv_b), conv_d3], dim=1))
        conv_u2 = self.conv_up2(torch.cat([self.upsamle(conv_u1), conv_d2], dim=1))
        conv_u3 = self.conv_up3(torch.cat([self.upsamle(conv_u2), conv_d1], dim=1))

        out = self.conv_out(conv_u3)
        return out