'''
unet网络需要的基本卷积网络模块
'''


import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''卷积模块，conv3*3=>batchnorm=>dropout=>ReLU'''

    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=3,padding=1),
            #nn.BatchNorm2d(mid_channels,momentum=0.12,track_running_stats=False),
            nn.InstanceNorm2d(mid_channels,affine=True),
            #momentum设置得过高可能导致BN层对数据变化过于敏感，而设置得低有利于处理噪音较多的数据
            #现在可能要放弃BN了，因为它对batchsize过小的训练影响太大了，而我batchsize又没法设置的很大
            #nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            #nn.BatchNorm2d(out_channels,momentum=0.12,track_running_stats=False),
            nn.InstanceNorm2d(out_channels,affine=True),
            #nn.Dropout(p=0.3),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class Down(nn.Module):
    '''maxpool with double conv'''

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    '''采用双线性插值/反卷积进行上采样'''
    def __init__(self,in_channels,out_channels,Transpose=True):
        super(Up,self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,2,stride=2)

        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                nn.Conv2d(in_channels,in_channels//2,kernel_size=1,padding=0),
                #nn.Dropout(p=0.2),
                nn.ReLU(inplace=True)
            )
        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        '''
            conv output shape = (input_shape-Fliter_shape+2*padding)/stride + 1
        '''

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1,(diffX // 2,diffX - diffX//2,
                                   diffY // 2,diffY - diffY//2))

        x = torch.cat([x2,x1],dim = 1)
        x= self.conv(x)
        return x



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)