import numpy as np
import torch
from torch import nn
from common_dl import device,init_weights,Tensor

## removed the batch_norm, use the LeakyReLU function
class deepnet(nn.Module):
    def __init__(self,gen_method,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num # class_num=1 for both CWGANGP and CDCGAN
        self.label_embedding = nn.Embedding(5, self.chn_num)
        self.wind=wind
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,50), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.LeakyReLU(0.2, inplace=True)
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)

        self.conv2=nn.Conv2d(64, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.LeakyReLU(0.2, inplace=True)
        self.drop2=nn.Dropout(p=0.5, inplace=False)

        self.conv3=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.LeakyReLU(0.2, inplace=True)
        self.drop3=nn.Dropout(p=0.5, inplace=False)

        self.conv4=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.LeakyReLU() # ELU
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.seq=nn.Sequential(self.conv_time,self.conv_spatial,self.nonlinear1,self.drop1,self.conv2,
                          self.nonlinear2,self.drop2,self.conv3,self.nonlinear3,self.drop3,self.conv4,
                          self.nonlinear4,self.drop4)
        #out = self.seq(np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32)))
        out = self.seq(torch.FloatTensor(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32)))
        len=out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=50, out_features=self.class_num, bias=True)
        self.layers = self.create_layers_field()
        self.apply(init_weights)

    def create_layers_field(self):
        layers = []
        for idx, m in enumerate(self.modules()):
            if (type(m) == nn.Conv2d or type(m) == nn.Linear):
                layers.append(m)
        return layers

    def forward(self,x, labels):
        lab_emb=self.label_embedding(labels).squeeze() # torch.Size([32, 10])
        # expect input shape: (batch_size, channel_number, time_length)
        x = torch.cat((x, lab_emb.unsqueeze(2)), -1) # emb: torch.Size([32,10])
        x=x.unsqueeze(1)
        y=self.seq(x)
        y=self.ap(y)
        y=y.squeeze()
        y=self.final_linear(y)
        return y
