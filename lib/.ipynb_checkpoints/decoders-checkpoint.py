import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

import math
from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc 
from .segFormer import *
from einops import rearrange
import sys

from lib.gcn_lib import Grapher as GCB 
from lib.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerBlock

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UCB(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1, activation='relu'):
        super(UCB,self).__init__()
        
        if(activation=='leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif(activation=='hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:    
            self.activation = nn.ReLU(inplace=True)
            
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups,bias=True),
	    nn.BatchNorm2d(ch_in),
	    self.activation,
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0,bias=True),
           )

    def forward(self,x):
        x = self.up(x)
        return x

class trans_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=4, stride=2, padding=1, groups=32):
        super(trans_conv,self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        activation = 'relu'
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            #nn.GroupNorm(1,1),
            nn.Sigmoid()
        )
        
        if(activation=='leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif(activation=='hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:    
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        activation='relu'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)
        
        if(activation=='leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif(activation=='hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:    
            self.activation = nn.ReLU(inplace=True)
            
        self.fc2   = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 
    
    # # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


class SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes * 2, stride)
        self.bn1 = nn.BatchNorm2d(inplanes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes * 2, inplanes * 2 * 2)
        self.bn2 = nn.BatchNorm2d(inplanes * 2 * 2)
        self.conv3 = conv3x3(inplanes * 2 * 2, inplanes * 2)
        self.bn3 = nn.BatchNorm2d(inplanes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # if planes <= 16:
        #     self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)  # (224, 300) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        # elif planes == 7:
        #     self.globalAvgPool = nn.AvgPool2d((7, 7), stride=1)  # (112, 150) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((7, 7), stride=1)
        # elif planes == 14:
        #     self.globalAvgPool = nn.AvgPool2d((14, 14), stride=1)    # (56, 75) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((14, 14), stride=1)
        # elif planes == 28:
        #     self.globalAvgPool = nn.AvgPool2d((28, 28), stride=1)    # (28, 37) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((28, 28), stride=1)
        # elif planes == 56:
        #     self.globalAvgPool = nn.AvgPool2d((56, 56), stride=1)    # (14, 18) for ISIC2018
        #     self.globalMaxPool = nn.MaxPool2d((56, 56), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        print(out.shape)
        sys.exit()
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight

class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)  

class SPAv2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPAv2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class CUP(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CUP,self).__init__()
        
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])

    def forward(self,x, skips):

        d4 = self.ConvBlock4(x)
        
        # decoding + concat path
        d3 = self.Up3(d4)
        d3 = torch.cat((skips[0],d3),dim=1)
        
        d3 = self.ConvBlock3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((skips[1],d2),dim=1)
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((skips[2],d1),dim=1)
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1              

class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SPA()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1       

class CASCADE(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SPA()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = d3 + x3
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = d2 + x2
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = d1 + x1
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1

class GCUP(nn.Module):
    def __init__(self, channels=[512,320,128,64], img_size=224, drop_path_rate=0.0, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCUP,self).__init__()
        
        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation
        
        #  Graph convolution block (GCB) parameters
        self.padding=padding
        self.k = k # neighbor num (default:9)
        self.conv = conv # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch' # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1,1,4, 2]
        self.dpr = [self.drop_path,self.drop_path,self.drop_path,self.drop_path]  # stochastic depth decay rule 
        self.num_knn = [self.k,self.k,self.k,self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4
        
        self.gcb4 = nn.Sequential(GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW//(4*4*4), drop_path=self.dpr[0],
                                    relative_pos=True, padding=self.padding),
        )
	
        self.ucb3 = UCB(ch_in=channels[0],ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW//(4*4), drop_path=self.dpr[1],
                                    relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1],ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW//(4), drop_path=self.dpr[2],
                                    relative_pos=True, padding=self.padding),
        )
        
        self.ucb1 = UCB(ch_in=channels[2],ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                    relative_pos=True, padding=self.padding),
        )
      
    def forward(self,x, skips):
        
        # GCAM4
        d4 = self.gcb4(x)        
        
        # UCB3
        d3 = self.ucb3(d4)
        
        # Aggregation 3
        d3 = d3 + skips[0]
        
        # GCAM3
        d3 = self.gcb3(d3)       
        
        # UCB2
        d2 = self.ucb2(d3)       
        
        # Aggregation 2
        d2 = d2 + skips[1] 
        
        # GCAM2
        d2 = self.gcb2(d2)
        
        # UCB1
        d1 = self.ucb1(d2)
                
        # Aggregation 1
        d1 = d1 + skips[2]
        
        # GCAM1
        d1 = self.gcb1(d1)
        
        return d4, d3, d2, d1

class GCUP_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64], img_size=224, drop_path_rate=0.0, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCUP_Cat,self).__init__()
        
        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation
        
        #  Graph convolution block (GCB) parameters
        self.padding=padding
        self.k = k # neighbor num (default:9)
        self.conv = conv # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch' # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1,1,4, 2]
        self.dpr = [self.drop_path,self.drop_path,self.drop_path,self.drop_path]  # stochastic depth decay rule 
        self.num_knn = [self.k,self.k,self.k,self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4
        
        self.gcb4 = nn.Sequential(GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW//(4*4*4), drop_path=self.dpr[0],
                                    relative_pos=True, padding=self.padding),
        )
	
        self.ucb3 = UCB(ch_in=channels[0],ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(GCB(2*channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW//(4*4), drop_path=self.dpr[1],
                                    relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=2*channels[1],ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(GCB(2*channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW//(4), drop_path=self.dpr[2],
                                    relative_pos=True, padding=self.padding),
        )
        
        self.ucb1 = UCB(ch_in=2*channels[2],ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(GCB(2*channels[3], self.num_knn[3],  min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                    relative_pos=True, padding=self.padding),
        )       
      
    def forward(self,x, skips):
        
        # GCAM4
        d4 = self.gcb4(x)         
        
        # UCB3
        d3 = self.ucb3(d4)

        # Aggregation 3
        d3 = torch.cat((skips[0],d3),dim=1)
        
        # GCAM3
        d3 = self.gcb3(d3)

        # UCB2
        d2 = self.ucb2(d3)

        # Aggregation 2
        d2 = torch.cat((skips[1],d2),dim=1)
        
        # GCAM2
        d2 = self.gcb2(d2)
        
        # UCB1
        d1 = self.ucb1(d2)
        
        # Aggregation 1
        d1 = torch.cat((skips[2],d1),dim=1)
        
        # GCAM1
        d1 = self.gcb1(d1)

        return d4, d3, d2, d1
    
    
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels, F_x=out_channels)
        self.nConvs = _make_nConv(in_channels + out_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        # up = self.up(x)
        up = x
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class GCASCADE(nn.Module):
    def __init__(self, channels=[512,320,128,64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCASCADE,self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation
        
        #  Graph convolution block (GCB) parameters
        self.padding=padding
        self.k = k # neighbor num (default:9)
        self.conv = conv # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch' # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1,1,4,2]
        self.dpr = [self.drop_path,self.drop_path,self.drop_path,self.drop_path]  # stochastic depth decay rule 
        self.num_knn = [self.k,self.k,self.k,self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4
        
        self.gcb4 = nn.Sequential(GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW//(4*4*4), drop_path=self.dpr[0],
                                    relative_pos=True, padding=self.padding),
        )
	
        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW//(4*4), drop_path=self.dpr[1],
                                    relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW//(4), drop_path=self.dpr[2],
                                    relative_pos=True, padding=self.padding),
        )
        
        self.ucb1 = UCB(ch_in=channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(GCB(channels[3], self.num_knn[3],  min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                    relative_pos=True, padding=self.padding),
        )

        # self.up4 = UpBlock_attention(channels[0]*2, channels[0], nb_Conv=2)
        # self.up3 = UpBlock_attention(channels[0], channels[1], nb_Conv=2)
        # self.up2 = UpBlock_attention(channels[1], channels[2], nb_Conv=2)
        # self.up1 = UpBlock_attention(channels[2], channels[3], nb_Conv=2)

        self.spa = SPA()

      
    def forward(self,x, skips):
        
        # d4 = self.up4(x, skips[0])
        
        # GCAM4
        d4 = self.gcb4(x) 
        d4 = self.spa(d4)*d4       
        
        # UCB3
        d3 = self.ucb3(d4)
        # d3 = self.up4(d4, skips[0])
        
        # Aggregation 3
        d3 = d3 + skips[0] #torch.cat((skips[0],d3),dim=1)
        # d3 = self.up3(d3, skips[1])
       
        # GCAM3
        d3 = self.gcb3(d3)
        d3 = self.spa(d3)*d3        
        
        # UCB2
        d2 = self.ucb2(d3)
        
        # Aggregation 2
        d2 = d2 + skips[1] #torch.cat((skips[1],d2),dim=1)
        # d2 = self.up2(d2, skips[2])
        
        # GCAM2
        d2 = self.gcb2(d2)
        d2 = self.spa(d2)*d2
        
        
        # UCB1
        d1 = self.ucb1(d2)
        
        # Aggregation 1
        d1 = d1 + skips[2] #torch.cat((skips[2],d1),dim=1)
        # d1 = self.up1(d1, skips[3])
        
        # GCAM1
        d1 = self.gcb1(d1)
        d1 = self.spa(d1)*d1
        
        return d4, d3, d2, d1

class GCASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCASCADE_Cat,self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation
        
        #  Graph convolution block (GCB) parameters
        self.padding=padding
        self.k = k # neighbor num (default:9)
        self.conv = conv # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch' # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1,1,4,2]
        self.dpr = [self.drop_path,self.drop_path,self.drop_path,self.drop_path]  # stochastic depth decay rule 
        self.num_knn = [self.k,self.k,self.k,self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4
        
        self.gcb4 = nn.Sequential(GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW//(4*4*4), drop_path=self.dpr[0],
                                    relative_pos=True, padding=self.padding),
        )
	
        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride = self.ucb_stride, padding = self.ucb_pad, groups = channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(GCB(channels[1]*2, self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW//(4*4), drop_path=self.dpr[1],
                                    relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1]*2, ch_out=channels[2], kernel_size=self.ucb_ks, stride = self.ucb_stride, padding = self.ucb_pad, groups = channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(GCB(channels[2]*2, self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW//(4), drop_path=self.dpr[2],
                                    relative_pos=True, padding=self.padding),
        )
        
        self.ucb1 = UCB(ch_in=channels[2]*2, ch_out=channels[3], kernel_size=self.ucb_ks, stride = self.ucb_stride, padding = self.ucb_pad, groups = channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(GCB(channels[3]*2, self.num_knn[3],  min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                    relative_pos=True, padding=self.padding),
        )        
        
        self.spa = SPA()

      
    def forward(self,x, skips):   
        
        # GCAM4
        d4 = self.gcb4(x) 
        d4 = self.spa(d4)*d4        
        
        # UCB3
        d3 = self.ucb3(d4)
        
        # Aggregation 3
        d3 = torch.cat((skips[0],d3),dim=1)
        
        # GCAM3
        d3 = self.gcb3(d3)
        d3 = self.spa(d3)*d3                
        
        # ucb2
        d2 = self.ucb2(d3)
        
        # Aggregation 2
        d2 = torch.cat((skips[1],d2),dim=1)
        
        # GCAM2
        d2 = self.gcb2(d2)
        d2 = self.spa(d2)*d2
        
        
        # ucb1
        d1 = self.ucb1(d2)
        
        # Aggregation 1
        d1 = torch.cat((skips[2],d1),dim=1)
        
        # GCAM1
        d1 = self.gcb1(d1)
        d1 = self.spa(d1)*d1
        
        return d4, d3, d2, d1
    
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x
    
class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, up_chan, heads, reduction_ratios,token_mlp_mode,resolution, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        self.resolution = resolution
        dims = in_out_chan[0]
        self.out_dim = out_dim = in_out_chan[1]
        self.up = UpBlock_attention(up_chan[0], up_chan[1], nb_Conv=2)
        self.spa = SPA()
        self.cpa = ChannelAttention(in_out_chan[1])
        self.ConvBlock3 = conv_block(ch_in=in_out_chan[1], ch_out=in_out_chan[1])
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            # self.last_layer = nn.Conv2d(out_dim, n_class,1)
            self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        tmp = x1
        if x2 is not None:
            b, c, h, w = x2.shape
            B, _, C = x1.shape
            x1 = x1.transpose(1, 2).reshape(B, C, h, w)
            # x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            # cat_x = torch.cat([x1, x2], dim=-1)
        
            cat_x = self.up(x1, x2)
            cat_x = cat_x.reshape(b, c, -1).transpose(1, 2);
            
            # print("-----catx shape", cat_x.shape)
            # cat_linear_x = self.concat_linear(cat_x)
            cat_linear_x = cat_x
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            tmp = tran_layer_2
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
            
            # print(out.shape)
        tmp = tmp.transpose(1,2).reshape(tmp.size(0), self.out_dim, self.resolution, self.resolution)
        tmp = self.spa(tmp)*tmp
        tmp = self.cpa(tmp)*tmp
        tmp = self.ConvBlock3(tmp)
        return tmp, out
        
