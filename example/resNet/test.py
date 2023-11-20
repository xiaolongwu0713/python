import torch
from torch import nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url

from common_dl import count_parameters
from torchvision.models.resnet import resnet18
from examples.resNet.models import  my_resnet18

mm=my_resnet18(6,5, pretrained=True)
x= torch.randn(1, 6, 125, 300)
mm(x).shape