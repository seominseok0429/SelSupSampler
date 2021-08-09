from __future__ import division
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pdb

cfg = {
    'VGG5' : [32,64,128,256],
    'VGG5_2' : [32,64,128,256],
}

class VGG(nn.Module):
    def __init__(self,vgg_name,num_class = 64):
        super(VGG,self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(128,num_class)

        self.conv1 = nn.Conv3d(3,8,kernel_size = (7,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn1 = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8,16,kernel_size = (7,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16,32,kernel_size = (7,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn3 = nn.BatchNorm3d(32)

        self.conv4 = nn.Conv3d(32,64,kernel_size = (7,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(64,128,kernel_size = (7,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn5 = nn.BatchNorm3d(128)

        self.conv6 = nn.Conv3d(256,512,kernel_size = (3,3,3), stride=(1,2,2),padding=(3,1,1),bias=False)
        self.bn6 = nn.BatchNorm3d(512)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(7,3,3),stride=(1,2,2),padding=(3,1,1))
        self.hi = nn.AdaptiveAvgPool3d((1,1,1))
    def _make_layers(self, cfg, stride = 1):

        layers = []
        in_channels = 8
        for x in cfg:
             layers += [nn.Conv3d(in_channels,x,kernel_size = (7,3,3) , stride = (1,2,2), padding = (3,1,1), bias = False),
                     nn.BatchNorm3d(x),
                     nn.ReLU(inplace=True)]
             in_channels = x
#            layers += [nn.Conv3d(in_channels,x,kernel_size = 3 , stride = 2, padding = 1, bias = False),
#                    nn.BatchNorm3d(x),
#                    nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool3d((1,1,1))]
        return nn.Sequential(*layers)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        #x = self.relu(self.bn6(self.conv6(x)))
        x = self.hi(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def vgg5(**kwargs):
    net = VGG('VGG5')
    return net
def vgg5_2(**kwargs):
    net = VGG('VGG5_2')

#net = vgg5()
#image = torch.randn(1,3,64,32,32)
#a = net(image)
#print(a.shape)
