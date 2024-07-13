import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage

from lib.pvtv2 import pvt_v2_b2, pvt_v2_b5, pvt_v2_b0
from lib.decoders import CUP, CASCADE, CASCADE_Cat, GCUP, GCUP_Cat, GCASCADE, GCASCADE_Cat, MyDecoderLayer
from lib.decoders import CUP, CASCADE, CASCADE_Cat, GCUP, GCUP_Cat, GCASCADE, GCASCADE_Cat
from lib.pyramid_vig import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu

from lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
from lib.vision_transformer import SwinUnet
from lib.CTrans import ChannelTransformer

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class PVT_CUP(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CUP, self).__init__()
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CUP(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('CUP decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

class PVT_CASCADE(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('CASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        print(p1.shape)
        sys.exit()
        return p1, p2, p3, p4

class PVT_CASCADE_Cat(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE_Cat, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print('Model %s created, param count: %d' %
                     ('PVT backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        # decoder initialization
        self.decoder = CASCADE_Cat(channels=[512, 320, 128, 64])
        
        print('Model %s created, param count: %d' %
                     ('CASCADE_Cat decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

                        
class PVT_GCUP(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(PVT_GCUP, self).__init__()
        
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCUP_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCUP_decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4  
                
class PVT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(PVT_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]

        self.mtc = ChannelTransformer(vis=False, img_size=img_size,
                                     channel_num=[self.channels[3], self.channels[2], self.channels[1], self.channels[0]],
                                     patchSize=[8,4,2,1])
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        #torch.Size([1, 64, 56, 56])
        #torch.Size([1, 128, 28, 28])
        #torch.Size([1, 320, 14, 14])
        #torch.Size([1, 512, 7, 7])
        x1,x2,x3,x4,att_weights = self.mtc(x1, x2, x3, x4)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Unet(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(Unet, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]

        self.mtc = ChannelTransformer(vis=False, img_size=img_size,
                                     channel_num=[self.channels[3], self.channels[2], self.channels[1], self.channels[0]],
                                     patchSize=[8,4,2,1])

        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        resolution = [56,28,14,7]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        up_chan = [[64,64],[160,128],[256,320],[512,512]]
        token_mlp_mode="mix_skip"

    
        # self.decoder_3= MyDecoderLayer((d_base_feat_size,d_base_feat_size), in_out_chan[3], up_chan[3], heads[3], reduction_ratios[3],token_mlp_mode,resolution[3], n_class=n_class)
        # self.decoder_2= MyDecoderLayer((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], up_chan[2], heads[2], reduction_ratios[2], token_mlp_mode,resolution[2], n_class=n_class)
        # self.decoder_1= MyDecoderLayer((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], up_chan[1], heads[1], reduction_ratios[1], token_mlp_mode,resolution[1], n_class=n_class)
        # self.decoder_0= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], up_chan[0], heads[0], reduction_ratios[0], token_mlp_mode,resolution[0], n_class=n_class, is_last=True)

        # scale atten
        # self.dec =  ChannelTransformer(vis=False, img_size=img_size,
        #                              channel_num=[self.channels[3], self.channels[2], self.channels[1], self.channels[0]],
        #                              patchSize=[8,4,2,1])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        B, C, H, W = x.shape
        resolution = [56, 28 ,14, 7]
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        x5 = x4
      
        #torch.Size([1, 64, 56, 56])
        #torch.Size([1, 128, 28, 28])
        #torch.Size([1, 320, 14, 14])
        #torch.Size([1, 512, 7, 7])
        x1,x2,x3,x4,att_weights = self.mtc(x1, x2, x3, x4)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        # b,c,_,_ = x4.shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------     
        # print("stage3-----")   
        # x4_o, tmp_3 = self.decoder_3(x4.permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")   
        # x3_o, tmp_2 = self.decoder_2(tmp_3, x3)
        # print("stage1-----")   
        # x2_o, tmp_1 = self.decoder_1(tmp_2, x2)
        # print("stage0-----")  
        # x1_o, tmp_0 = self.decoder_0(tmp_1, x1)
        # x4_o = x4_o.transpose(1, 2).view(B, 512, resolution[3], resolution[3])
        # x3_o = x3_o.transpose(1, 2).view(B, 320, resolution[2], resolution[2])
        # x2_o = x2_o.transpose(1, 2).view(B, 128, resolution[1], resolution[1])
        # x1_o = x1_o.transpose(1, 2).view(B, 64, resolution[0], resolution[0])

        # x4_o, x3_o, x2_o, x1_o, att_weights_tmp = self.dec(x1_o, x2_o, x3_o, x4_o)
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4
    
class PVT_UPGCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(PVT_UPGCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]

        self.mtc = ChannelTransformer(vis=False, img_size=img_size,
                                     channel_num=[self.channels[3], self.channels[2], self.channels[1], self.channels[0]],
                                     patchSize=[8,4,2,1])

        
        # decoder initialization
#         if self.skip_aggregation == 'additive':
#             self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
#         elif self.skip_aggregation == 'concatenation':
#             self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
#             self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
#         else:
#             print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
#             self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

#         print('Model %s created, param count: %d' %
#                      ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        resolution = [56,28,14,7]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        up_chan = [[64,64],[160,128],[256,320],[512,512]]
        token_mlp_mode="mix_skip"

    
        self.decoder_3= MyDecoderLayer((d_base_feat_size,d_base_feat_size), in_out_chan[3], up_chan[3], heads[3], reduction_ratios[3],token_mlp_mode,resolution[3], n_class=n_class)
        self.decoder_2= MyDecoderLayer((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], up_chan[2], heads[2], reduction_ratios[2], token_mlp_mode,resolution[2], n_class=n_class)
        self.decoder_1= MyDecoderLayer((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], up_chan[1], heads[1], reduction_ratios[1], token_mlp_mode,resolution[1], n_class=n_class)
        self.decoder_0= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], up_chan[0], heads[0], reduction_ratios[0], token_mlp_mode,resolution[0], n_class=n_class, is_last=True)

        # scale atten
        self.dec =  ChannelTransformer(vis=False, img_size=img_size,
                                     channel_num=[self.channels[3], self.channels[2], self.channels[1], self.channels[0]],
                                     patchSize=[8,4,2,1])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        B, C, H, W = x.shape
        resolution = [56, 28 ,14, 7]
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        x5 = x4
      
        #torch.Size([1, 64, 56, 56])
        #torch.Size([1, 128, 28, 28])
        #torch.Size([1, 320, 14, 14])
        #torch.Size([1, 512, 7, 7])
        x1,x2,x3,x4,att_weights = self.mtc(x1, x2, x3, x4)

        # decoder
        # x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        b,c,_,_ = x4.shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------     
        # print("stage3-----")   
        x4_o, tmp_3 = self.decoder_3(x4.permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")   
        x3_o, tmp_2 = self.decoder_2(tmp_3, x3)
        # print("stage1-----")   
        x2_o, tmp_1 = self.decoder_1(tmp_2, x2)
        # print("stage0-----")  
        x1_o, tmp_0 = self.decoder_0(tmp_1, x1)
        # x4_o = x4_o.transpose(1, 2).view(B, 512, resolution[3], resolution[3])
        # x3_o = x3_o.transpose(1, 2).view(B, 320, resolution[2], resolution[2])
        # x2_o = x2_o.transpose(1, 2).view(B, 128, resolution[1], resolution[1])
        # x1_o = x1_o.transpose(1, 2).view(B, 64, resolution[0], resolution[0])

        x4_o, x3_o, x2_o, x1_o, att_weights_tmp = self.dec(x1_o, x2_o, x3_o, x4_o)
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4


class Swin_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(Swin_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = SwinUnet(num_classes=n_class)  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [768, 384, 192, 96]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):

        resolution = [56, 28 ,14, 7]
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        B, L, C = x1.shape
        x1 = x1.transpose(1, 2).view(B, C, resolution[0], resolution[0])

        B, L, C = x2.shape
        x2 = x2.transpose(1, 2).view(B, C, resolution[1], resolution[1])

        B, L, C = x3.shape
        x3 = x3.transpose(1, 2).view(B, C, resolution[2], resolution[2])

        B, L, C = x4.shape
        x4 = x4.transpose(1, 2).view(B, C, resolution[3], resolution[3])
        #torch.Size([1, 56*56, 96])
        #torch.Size([1, 28*28, 192])
        #torch.Size([1, 14*14, 384])
        #torch.Size([1, 7*7, 768])
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4
    

        

class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.

    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x

        
class MERIT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', interpolation='bilinear', skip_aggregation='additive'):
        super(MERIT_GCASCADE, self).__init__()
        
        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
  
        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        # state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')        
        # self.backbone1.load_state_dict(state_dict1, strict=False)
        
        print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')       
        # state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')        
        # self.backbone2.load_state_dict(state_dict2, strict=False)
        
        print('Pretrain weights loaded.')
        
        self.channels = [768, 384, 192, 96]
        
        # decoder initialization 
        if self.skip_aggregation == 'additive':
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder1 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
               
        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder1.parameters()])))
        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder2.parameters()])))
                         
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
            
        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))                
        print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
        sys.exit()
        
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder1(f1[3], [f1[2], f1[1], f1[0]])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)
        

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)        
        x_in = x * p14_in
                
        f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
                    
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        x21_o, x22_o, x23_o, x24_o = self.decoder2(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        #print([p21.shape,p22.shape,p23.shape,p24.shape])
               
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p1, p2, p3, p4
                                
if __name__ == '__main__':
    model = PVT_GCASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

