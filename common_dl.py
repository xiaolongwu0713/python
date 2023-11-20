import os
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import random
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu') # ad-hoc
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if (type(m) == nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

# convert input of (batch, height, width) to (batch, channel, height, width)
class add_channel_dimm(torch.nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(1)
        return x


class squeeze_all(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel() # numel: return how many number in that parameter
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def parameterNum(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
class myDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x_tensor, y_tensor,norm=False):
        self.x = x_tensor
        self.y = y_tensor
        self.norm=norm
        assert len(self.x) == len(self.y)
    def __getitem__(self, index):
        if self.norm:
            tmp=scaler.fit_transform(self.x[index].transpose())
            return torch.from_numpy(tmp.transpose()), self.y[index]
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

def collate_fn(data):
    data=[dataa[0].transpose(1,0) for dataa in data]
    # new_data=[(dataa,500) for dataa in data] #list of tuple of (data,window_size)
    new_data=data # list of data
    new_data=torch.from_numpy(np.asarray(new_data))
    return new_data,'dumm'

class myDataset_timeGan(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x_tensor,y_tensor,feat_dim=500):
        self.x = x_tensor
        self.feat_dim = feat_dim
        assert self.x.shape[-1] == self.feat_dim
    def __getitem__(self, index):
        return self.x[index], self.feat_dim
    def __len__(self):
        return self.x.shape[0]






