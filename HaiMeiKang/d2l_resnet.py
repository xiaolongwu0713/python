import torch
from torch import nn
from torch.nn import functional as F
#from d2l import torch as d2l

from common_dl import add_channel_dimm, squeeze_all


class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        #self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()
        # keep width and hight by using kernel_size=3 and pedding=1;
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = self.activation(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.activation(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class d2lresnet(nn.Module):
    def __init__(self,task='classification', class_num=5, reg_d=23,end_with_logsoftmax=False,channel_num=1): # as_DA_discriminator=False
        super().__init__()
        if channel_num==1:
            maxpool=nn.MaxPool2d(kernel_size=(1,3), stride=(1,1))
        else:
            maxpool=nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.end_with_logsoftmax=end_with_logsoftmax
        self.activation = nn.ReLU()
        self.task=task
        if self.task=='classification':
            self.target_d=class_num
        elif self.task=='regression':
            self.target_d=reg_d
        #self.block_num=block_num
        b1 = nn.Sequential(add_channel_dimm(),nn.Conv2d(1, 64, kernel_size=(1,5), stride=(1,1)), # kernel_size=(1,50),
                           nn.BatchNorm2d(64), self.activation,
                           maxpool) # shape: (batch, channel/plan, electrode, time)
        b2 = nn.Sequential(*resnet_block(64, 64, 1, first_block=True))# shape: (batch, channel/plan, electrode, time)
        b3 = nn.Sequential(*resnet_block(64, 128, 1))# shape: (batch, channel/plan, electrode, time)
        b4 = nn.Sequential(*resnet_block(128, 256, 1))
        b5 = nn.Sequential(*resnet_block(256, 512, 1))
        b6 = nn.Sequential(*resnet_block(512, 1024, 1))
        b7 = nn.Sequential(*resnet_block(1024, 2048, 1))

        self.d2lresnet = nn.Sequential(b1, b2, b3, b4, b5,b6,b7, nn.AdaptiveAvgPool2d((1, 1)),squeeze_all())
        self.dropoutAndlinear=nn.Sequential(nn.Dropout(p=0.5),nn.Linear(2048, 1024),self.activation
                                            ,nn.Linear(1024, self.target_d))


    def forward(self,x): #x:torch.Size(batch, channel, time])
        y=self.d2lresnet(x) # use CrossEntropyLoss loss
        y=self.dropoutAndlinear(y)
        if self.end_with_logsoftmax==True: #self.as_discriminator==False:
            return F.log_softmax(y, dim=1) # use a NLLLoss
        else:
            return y #use a torch.nn.CrossEntropyLoss

    #x=torch.randn(32,1,208,500)






