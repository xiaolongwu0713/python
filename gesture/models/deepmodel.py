import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from torch.nn import functional as F

from braindecode.models.modules import Expression, AvgPool2dWithConv, Ensure4d
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output
from braindecode.util import np_to_var
from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net

from common_dl import add_channel_dimm,init_weights
from gesture.models.utils import squeeze_all


class expand_dim(torch.nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(1)
        return x

def swap_time_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 1, 3, 2)

def swap_plan_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0,2,1,3)

#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_rnn(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num
        self.wind=wind
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,50), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.ELU()
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)
        self.conv2=nn.Conv2d(64, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.ELU()
        self.drop2=nn.Dropout(p=0.5, inplace=False)
        self.conv3=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.ELU()
        self.drop3=nn.Dropout(p=0.5, inplace=False)
        self.conv4=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.ELU()
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.seq=nn.Sequential(self.conv_time,self.conv_spatial,self.bn1,self.nonlinear1,self.mp1,self.drop1,self.conv2,self.bn2,
                          self.nonlinear2,self.drop2,self.conv3,self.bn3,self.nonlinear3,self.drop3,self.conv4,self.bn4,
                          self.nonlinear4,self.drop4)
        out = self.seq(np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32))) #(32,50,1,123)
        len=out.shape[3]
        Hin=out.shape[1]

        self.lstm1 = nn.LSTM(Hin, int(Hin/2), batch_first=True)
        self.drop5 = nn.Dropout(p=0.2, inplace=False)
        #self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=int(Hin/2), out_features=self.class_num, bias=True)

    def forward(self,x):
        # expect input shape: (batch_size, channel_number, time_length)
        x=x.unsqueeze(1)
        y=self.seq(x)
        y = y.squeeze(dim=2)
        y = y.permute(0, 2, 1)
        y, _ = self.lstm1(y)
        y = y[:, -1, :]
        #y=self.ap(y)
        #y=y.squeeze()
        y=self.drop5(y)
        y=self.final_linear(y)
        return y


#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
# standard scaler is very important for the deepnet model
class deepnet(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num
        self.wind=wind
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,50), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.ELU()
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)

        self.conv2=nn.Conv2d(64, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.ELU()
        self.drop2=nn.Dropout(p=0.5, inplace=False)

        self.conv3=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.ELU()
        self.drop3=nn.Dropout(p=0.5, inplace=False)

        self.conv4=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.ELU()
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.seq=nn.Sequential(self.conv_time,self.conv_spatial,self.bn1,self.nonlinear1,self.mp1,self.drop1,self.conv2,self.bn2,
                          self.nonlinear2,self.drop2,self.conv3,self.bn3,self.nonlinear3,self.drop3,self.conv4,self.bn4,
                          self.nonlinear4,self.drop4)
        out = self.seq(np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32)))
        len=out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=50, out_features=self.class_num, bias=True)
        self.sigm=nn.Sigmoid() #
        self.layers = self.create_layers_field()
        self.apply(init_weights)

    def create_layers_field(self):
        layers = []
        for idx, m in enumerate(self.modules()):
            if (type(m) == nn.Conv2d or type(m) == nn.Linear):
                layers.append(m)
        return layers

    def forward(self,x):
        # expect input shape: (batch_size, channel_number, time_length)
        x=x.unsqueeze(1)
        y=self.seq(x)
        y=self.ap(y)
        y=y.squeeze()
        y=self.final_linear(y)
        return y

#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_expandPlan(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num
        self.wind=wind
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,10), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.ELU()
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)

        self.conv2=nn.Conv2d(64, 128, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.ELU()
        self.drop2=nn.Dropout(p=0.5, inplace=False)

        self.conv3=nn.Conv2d(128, 256, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.ELU()
        self.drop3=nn.Dropout(p=0.5, inplace=False)

        self.conv4=nn.Conv2d(256, 512, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.ELU()
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.seq=nn.Sequential(self.conv_time,self.conv_spatial,self.bn1,self.nonlinear1,self.mp1,self.drop1,self.conv2,self.bn2,
                          self.nonlinear2,self.drop2,self.conv3,self.bn3,self.nonlinear3,self.drop3,self.conv4,self.bn4,
                          self.nonlinear4,self.drop4)
        out = self.seq(np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32)))
        len=out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=512, out_features=self.class_num, bias=True)

    def forward(self,x):
        # expect input shape: (batch_size, channel_number, time_length)
        x=x.unsqueeze(1)
        y=self.seq(x)
        y=self.ap(y)
        y=y.squeeze()
        y=self.final_linear(y)
        return y


# add another spatial filter at the top of the net
#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_da(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num
        self.wind=wind
        self.conv_spatial_top = nn.Conv2d(1, self.chn_num, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,50), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.ELU()
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)
        self.conv2=nn.Conv2d(64, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.ELU()
        self.drop2=nn.Dropout(p=0.5, inplace=False)
        self.conv3=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.ELU()
        self.drop3=nn.Dropout(p=0.5, inplace=False)
        self.conv4=nn.Conv2d(50, 50, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.ELU()
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.bn0=nn.BatchNorm2d(self.chn_num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear0=nn.ELU()
        self.drop0=nn.Dropout(p=0.5, inplace=False)
        #self.seq0=nn.Sequential(self.bn0,self.nonlinear0,self.drop0)
        self.seq0 = nn.Sequential(self.drop0)

        self.seq = nn.Sequential(Expression(swap_plan_spat),self.conv_time, self.drop0, self.conv_spatial, self.bn1, self.nonlinear1, self.mp1,
                                 self.drop1, self.conv2, self.bn2, self.nonlinear2, self.drop2, self.conv3, self.bn3,
                                 self.nonlinear3, self.drop3, self.conv4, self.bn4, self.nonlinear4, self.drop4)

        test=np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32))
        test=self.conv_spatial_top(test)
        test=self.seq0(test)
        #test=test.permute(0,2,1,3)
        out=self.seq(test)
        len=out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=50, out_features=self.class_num, bias=True)

    def forward(self,x):
        # expect input shape: (batch_size, channel_number, time_length)
        x=x.unsqueeze(1)
        x = self.conv_spatial_top(x)
        x = self.seq0(x)
        #x = x.permute(0, 2, 1, 3)
        y=self.seq(x)
        y=self.ap(y)
        y=y.squeeze()
        y=self.final_linear(y)
        return y

class deepnet_da_expandPlan(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        self.chn_num=chn_num
        self.class_num=class_num
        self.wind=wind
        self.conv_spatial_top = nn.Conv2d(1, self.chn_num, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.conv_time=nn.Conv2d(1, 64, kernel_size=(1,10), stride=(1, 1))
        self.conv_spatial=nn.Conv2d(64, 64, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1=nn.ELU()
        self.mp1=nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=0, dilation=1, ceil_mode=False)
        self.drop1=nn.Dropout(p=0.5, inplace=False)
        self.conv2=nn.Conv2d(64, 128, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn2=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear2=nn.ELU()
        self.drop2=nn.Dropout(p=0.5, inplace=False)
        self.conv3=nn.Conv2d(128, 256, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn3=nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear3=nn.ELU()
        self.drop3=nn.Dropout(p=0.5, inplace=False)
        self.conv4=nn.Conv2d(256, 512, kernel_size=(1,10), stride=(1, 1), bias=False)
        self.bn4=nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear4=nn.ELU()
        self.drop4=nn.Dropout(p=0.5, inplace=False)

        self.bn0=nn.BatchNorm2d(self.chn_num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear0=nn.ELU()
        self.drop0=nn.Dropout(p=0.5, inplace=False)
        #self.seq0=nn.Sequential(self.bn0,self.nonlinear0,self.drop0)
        self.seq0 = nn.Sequential(self.drop0)

        self.seq = nn.Sequential(Expression(swap_plan_spat),self.conv_time, self.drop0, self.conv_spatial, self.bn1, self.nonlinear1, self.mp1,
                                 self.drop1, self.conv2, self.bn2, self.nonlinear2, self.drop2, self.conv3, self.bn3,
                                 self.nonlinear3, self.drop3, self.conv4, self.bn4, self.nonlinear4, self.drop4)

        test=np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32))
        test=self.conv_spatial_top(test)
        test=self.seq0(test)
        #test=test.permute(0,2,1,3)
        out=self.seq(test)
        len=out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear=nn.Linear(in_features=512, out_features=self.class_num, bias=True)

    def forward(self,x):
        # expect input shape: (batch_size, channel_number, time_length)
        x=x.unsqueeze(1)
        x = self.conv_spatial_top(x)
        x = self.seq0(x)
        #x = x.permute(0, 2, 1, 3)
        y=self.seq(x)
        y=self.ap(y)
        y=y.squeeze()
        y=self.final_linear(y)
        return y


# model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_changeDepth(nn.Module):
    def __init__(self, chn_num, class_num, wind, blocks):
        super().__init__()
        self.blocks=blocks
        self.chn_num = chn_num
        self.class_num = class_num
        self.wind = wind
        self.conv_time = nn.Conv2d(1, 64, kernel_size=(1, 50), stride=(1, 1))
        self.conv_spatial = nn.Conv2d(64, 50, kernel_size=(self.chn_num, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.nonlinear1 = nn.ELU()
        self.mp1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)

        self.seq=nn.Sequential(nn.Conv2d(50, 50, kernel_size=(1, 10), stride=(1, 1), bias=False),
        nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ELU(), nn.Dropout(p=0.5, inplace=False))

        self.block_seq=[]
        for b in range(self.blocks):
            self.block_seq.append([])
            self.block_seq[b]=self.seq

        self.seq = nn.Sequential(self.conv_time, self.drop0,self.conv_spatial, self.bn1, self.nonlinear1, self.mp1, self.drop1,
                                 *self.block_seq)
        out = self.seq(np_to_var(np.ones((1, 1, self.chn_num, self.wind), dtype=np.float32)))
        len = out.shape[3]

        self.ap = nn.AvgPool2d(kernel_size=(1, len), stride=(1, len), padding=0)
        self.final_linear = nn.Linear(in_features=50, out_features=self.class_num, bias=True)

    def forward(self, x):
        # expect input shape: (batch_size, channel_number, time_length)
        x = x.unsqueeze(1)
        y = self.seq(x)
        y = self.ap(y)
        y = y.squeeze()
        y = self.final_linear(y)
        return y



# net = deepnet(n_chans,class_number,input_window_samples=wind,final_conv_length='auto',) # 81%
# expect input shape: (batch_size, channel_number, time_length)
class deepnet_seq(nn.Sequential):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self,in_chans,n_classes,input_window_samples,final_conv_length,n_filters_time=64,n_filters_spat=64,filter_time_length=50,
        pool_time_length=3,pool_time_stride=3,n_filters_2=50,filter_length_2=10,n_filters_3=50,filter_length_3=10,n_filters_4=50,
        filter_length_4=10,first_nonlin=elu,first_pool_mode="max",first_pool_nonlin=identity,later_nonlin=elu,later_pool_mode="max",
        later_pool_nonlin=identity,drop_prob=0.5,double_time_convs=False,split_first_layer=True,batch_norm=True,batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        self.add_module("Add_dim", expand_dim())
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(swap_time_spat))
            self.add_module("conv_time",nn.Conv2d(1,self.n_filters_time,(self.filter_time_length, 1),stride=1,),)
            self.add_module("conv_spat",nn.Conv2d(self.n_filters_time,self.n_filters_spat,(1, self.in_chans),
                                                  stride=(conv_stride, 1),bias=not self.batch_norm,),)
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module("conv_time",nn.Conv2d(self.in_chans,self.n_filters_time,(self.filter_time_length, 1),
                                                  stride=(conv_stride, 1),bias=not self.batch_norm,),)
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module("bnorm",nn.BatchNorm2d(n_filters_conv,momentum=self.batch_norm_alpha,affine=True,eps=1e-5,),)
        self.add_module("conv_nonlin", Expression(self.first_nonlin)) #elu
        self.add_module("pool",first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)),) #MaxPool2d
        #self.add_module("pool_nonlin", Expression(self.first_pool_nonlin)) # identity

        def add_conv_pool_block(model, n_filters_before, n_filters, filter_length, block_nr):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            self.add_module("conv" + suffix,nn.Conv2d(n_filters_before,n_filters,(filter_length, 1),
                    stride=(conv_stride, 1),bias=not self.batch_norm,),)
            if self.batch_norm:
                self.add_module("bnorm" + suffix,nn.BatchNorm2d(n_filters,momentum=self.batch_norm_alpha,affine=True,eps=1e-5,),)
            self.add_module("nonlin" + suffix, Expression(self.later_nonlin)) # elu

            # maxpool2d
            #self.add_module("pool" + suffix,later_pool_class(kernel_size=(self.pool_time_length, 1),stride=(pool_stride, 1),),)

            #Expression(expression=identity)
            #self.add_module("pool_nonlin" + suffix, Expression(self.later_pool_nonlin)) # identity

        add_conv_pool_block(self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2)
        add_conv_pool_block(self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3)
        add_conv_pool_block(self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4)

        self.add_module("last_drop", nn.Dropout(p=self.drop_prob))

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(np_to_var(np.ones((1, self.in_chans, self.input_window_samples),dtype=np.float32,)))
            n_channels=out.cpu().data.numpy().shape[1]
            n_out_time,n_out_spatial = out.cpu().data.numpy().shape[2],out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        #self.add_module("conv_classifier",nn.Conv2d(self.n_filters_4,self.n_classes,(self.final_conv_length, 1),bias=True,),)
        self.add_module("globalAvgPooling",nn.AvgPool2d((n_out_time,n_out_spatial)))
        self.add_module("squeeze1", Expression(squeeze_all))
        self.add_module("conv_classifier", nn.Linear(n_channels,n_classes))
        #self.add_module("softmax", nn.LogSoftmax(dim=1))
        #self.add_module("squeeze2", Expression(squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()


def shortcut(block):  # @save
    channels = [(i.in_channels, i.out_channels) for i in block if type(i) == nn.Conv2d]
    input_channels = channels[0][0]
    num_channels = channels[0][1]
    strides = [i.stride for i in block if type(i) == nn.Conv2d]
    stride = strides[0][0] * strides[0][1]
    # 1,pad when using non-1 sized kernel; 2, make sure the shortcut stride = conv stride
    if (not input_channels == num_channels) or (not stride == 1):  # when channel number or width/height changed
        _shortcut = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
    else:
        _shortcut = Expression(expression=identity)
    return _shortcut

# model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_resnet(nn.Module):
    def __init__(self,in_chans,n_classes,input_window_samples,n_filters_time=64,n_filters_spat=64,expand=True,
        filter_time_length=50,drop_prob=0.5,pool_time_length=3,pool_time_stride=3,n_filters_2=50,filter_length_2=10,n_filters_3=50,
        filter_length_3=10,n_filters_4=50,filter_length_4=10,first_nonlin=elu,first_pool_mode="max",first_pool_nonlin=identity,
        later_nonlin=elu,later_pool_mode="max",later_pool_nonlin=identity,double_time_convs=False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.drop_prob = drop_prob
        if expand==True:
            conv_channels = [64,128,256,512,1024]
        else:
            conv_channels = [64, 50,50,50,50]
            #conv_channels = [64, 64, 64, 64, 64]

        self.block0 = nn.Sequential(add_channel_dimm(),
                                    nn.Conv2d(1,self.n_filters_time,(1,self.filter_time_length),stride=1,),
                                    nn.Conv2d(self.n_filters_time,self.n_filters_spat,(self.in_chans,1),stride=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters_spat,affine=True,eps=1e-5,),
                                    nn.ELU(),
                                    nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))

        kernel=11
        padding=11//2
        stride=2
        self.block1 = nn.Sequential(nn.Dropout(p=self.drop_prob),
                           nn.Conv2d(conv_channels[0],conv_channels[1], (1, kernel),padding=(0,padding), stride=(1,stride)))

        self.block2=nn.Sequential(nn.BatchNorm2d(conv_channels[1], affine=True, eps=1e-5, ),
                             nn.ELU(),
                             nn.Dropout(p=self.drop_prob),
                             nn.Conv2d(conv_channels[1],conv_channels[2], (1,kernel),padding=(0,padding),stride=(1,stride))
                             )

        self.block3 = nn.Sequential(nn.BatchNorm2d(conv_channels[2], affine=True, eps=1e-5, ),
                               nn.ELU(),
                               nn.Dropout(p=self.drop_prob),
                               nn.Conv2d(conv_channels[2],conv_channels[3], (1, kernel),padding=(0,padding), stride=(1, stride))
                               )

        self.block4 = nn.Sequential(nn.BatchNorm2d(conv_channels[3], affine=True, eps=1e-5, ),
                               nn.ELU(),
                               nn.Dropout(p=self.drop_prob),
                               nn.Conv2d(conv_channels[3],conv_channels[4], (1, kernel),padding=(0,padding), stride=(1, stride))
                               )

        self.sc1 = shortcut(self.block1)
        self.sc2 = shortcut(self.block2)
        self.sc3 = shortcut(self.block3)
        self.sc4 = shortcut(self.block4)

        x = torch.randn(1, in_chans, input_window_samples)
        all=nn.Sequential(self.block0,self.block1,self.block2,self.block3,self.block4)
        out=all(x)
        out_channels,n_out_time,n_out_spatial = out.shape[1],out.shape[2], out.shape[3]


        #self.add_module("squeeze2", Expression(squeeze_final_output))
        self.block_final = nn.Sequential(nn.AvgPool2d((n_out_time,n_out_spatial)),
                                         Expression(squeeze_all),
                                         nn.Linear(out_channels,n_classes))
        self.softmax = nn.LogSoftmax()

        self.apply(init_weights)

        # Start in eval mode
        self.eval()



    def forward(self, x):
        x = self.block0(x) #torch.Size([32, 64, 1, 150])

        y = self.block1(x) #torch.Size([32, 50, 11, 75])
        ff= self.sc1(x)
        x=y+ff

        y = self.block2(x)
        ff= self.sc2(x)
        x = y + ff

        y = self.block3(x)
        ff= self.sc3(x)
        x = y + ff

        y = self.block4(x)
        ff= self.sc4(x)
        x = y + ff

        x = self.block_final(x)

        return self.softmax(x)

# not so good. only 65% for sid=10
class TSception2(nn.Module):
    def __init__(self, sampling_rate, chnNum, num_T, num_S,dropout):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception2, self).__init__()
        # try to use shorter conv kernel to capture high frequency
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        win = [int(tt * sampling_rate) for tt in self.inception_window]
        # [500, 250, 125, 62, 31, 15, 7]
        # in order to have a same padding, all kernel size shoule be even number, then stride=kernel_size/2
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation

        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[0]), stride=1, padding=(0, 250)),  # kernel: 500
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[1]), stride=1, padding=(0, 125)),  # 250
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[2] + 1), stride=1, padding=(0, 63)),  # kernel: 126
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[3]), stride=1, padding=(0, 31)),  # kernel:62
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[4] + 1), stride=1, padding=(0, 16)),  # 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception6 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[5] + 1), stride=1, padding=(0, 8)),  # 15
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception7 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[6] + 1), stride=1, padding=(0, 4)),  # 7
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5 * 0.5), 1),
                      stride=(int(chnNum * 0.5 * 0.5), 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_S * 6)
        self.BN_s = nn.BatchNorm2d(num_S * 6)

        self.drop = nn.Dropout(dropout)
        self.avg = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.lstm1 = nn.LSTM(90, 45, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(45, 15),
            nn.ReLU())

    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        self.float()
        x = torch.squeeze(x, dim=0)
        batch_size=x.shape[0]
        # y1 = self.Tception1(x)
        if (len(x.shape) < 4):
            x=torch.unsqueeze(x,dim=1)
        y2 = self.Tception2(x)
        y3 = self.Tception3(x)
        y4 = self.Tception4(x)
        y5 = self.Tception5(x)
        y6 = self.Tception6(x)
        y7 = self.Tception7(x)  # (batch_size, plan, channel, time)
        out = torch.cat((y2, y3, y4, y5, y6, y7), dim=1)  # concate alone plan
        #out = self.BN_t(out) #Todo: braindecode didn't use normalization between t and s filter.

        z1 = self.Sception1(out)
        z2 = self.Sception2(out)
        z3 = self.Sception2(out)
        out_final = torch.cat((z1, z2, z3), dim=2)
        out = self.BN_s(out_final)

        # Todo: test the effect of log(power)
        out = torch.pow(out, 2) # ([28, 18, 5, 15])
        # Braindecode use avgpool2d here
        # out = self.avg(out)
        out = torch.log(out)
        # Todo: test if drop is beneficial
        out = self.drop(out)

        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(batch_size, seqlen, input_size)  # ([280, 38, 21])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:, -1, :]))
        return pred


class MSFBCNN(nn.Module):
    def __init__(self, input_dim, output_dim, FT=10):
        super(MSFBCNN, self).__init__()
        self.T = input_dim[1]
        self.FT = FT
        self.D = 1
        self.FS = self.FT * self.D
        self.C = input_dim[0]
        self.output_dim = output_dim

        # Parallel temporal convolutions
        self.conv1a = nn.Conv2d(1, self.FT, (1, 65), padding=(0, 32), bias=False)
        self.conv1b = nn.Conv2d(1, self.FT, (1, 41), padding=(0, 20), bias=False)
        self.conv1c = nn.Conv2d(1, self.FT, (1, 27), padding=(0, 13), bias=False)
        self.conv1d = nn.Conv2d(1, self.FT, (1, 17), padding=(0, 8), bias=False)

        self.batchnorm1 = nn.BatchNorm2d(4 * self.FT, False)

        # Spatial convolution
        self.conv2 = nn.Conv2d(4 * self.FT, self.FS, (self.C, 1), padding=(0, 0), groups=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.FS, False)

        # Temporal average pooling
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15), padding=(0, 0))

        self.drop = nn.Dropout(0.5)

        result = self.test_dim(torch.ones(1, 1, self.C, self.T))
        # Classification
        # self.fc1 = nn.Linear(self.FS*math.ceil(1+(self.T-75)/15), self.output_dim)
        self.fc1 = nn.Linear(result.shape[1], self.output_dim)

    def test_dim(self, x):
        # Layer 1
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x3 = self.conv1c(x)
        x4 = self.conv1d(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.batchnorm1(x)

        # Layer 2
        x = torch.pow(self.batchnorm2(self.conv2(x)), 2)
        x = self.pooling2(x)
        x = torch.log(x)
        x = self.drop(x)

        # FC Layer
        x = x.view(-1, self.num_flat_features(x))
        return x

    def forward(self, x):
        # ensure 4-D for input as (batch, channel_number, time)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, dim=1)
        # Layer 1
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x3 = self.conv1c(x)
        x4 = self.conv1d(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.batchnorm1(x)

        # Layer 2
        x = torch.pow(self.batchnorm2(self.conv2(x)), 2)
        x = self.pooling2(x)
        x = torch.log(x)
        x = self.drop(x)

        # FC Layer
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


