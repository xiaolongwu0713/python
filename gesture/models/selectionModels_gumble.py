import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from gesture.models.deepmodel import deepnet
from example.gumbelSelection.ChannelSelection.models import  MSFBCNN

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
class SelectionLayer(nn.Module):
    def __init__(self, N, M, temperature=1.0):
        super(SelectionLayer, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.N = N
        self.M = M
        self.qz_loga = Parameter(torch.randn(N, M) / 100)  # ~N(0,1)

        self.temperature = self.floatTensor([temperature])
        self.freeze = False
        self.thresh = 10.0
        self.H=0.0

    #Sampling over Gumbel distri was done using Inverse transform sampling:1,sample random y from uniform distr;
    # 2,calcuate x from y using inverse CDF function;
    # standard Gumbel CDF:y=exp(-exp(-x))-->inverse function:x=-ln(-ln(y));
    # below is a concrete distribution
    def quantile_concrete(self, x):  # eq: 2
        g = -torch.log(-torch.log(x))  # gumbel distribution
        #g=torch.zeros(x.shape).cuda() # test no sampling: failed.
        y = (self.qz_loga + g) / self.temperature  # beta # torch.Size([208, 10]) + torch.Size([32, 208, 10])
        y = torch.softmax(y, dim=1)  # concrete distribution. torch.Size([16, 44, 3])
        return y

    def get_eps(self, size):
        eps = self.floatTensor(size).uniform_(epsilon, 1 - epsilon)
        return eps

    def sample_z(self, batch_size, training):
        if training:
            eps = self.get_eps(self.floatTensor(batch_size, self.N, self.M))
            z = self.quantile_concrete(eps)  # eq: 2: same shape as eps;  torch.Size([16, 44, 3])
            z = z.view(z.size(0), 1, z.size(1), z.size(2))  # torch.Size([16, 1, 44, 3])
            return z
        else: # freeze or evaluating
            ind = torch.argmax(self.qz_loga, dim=0)
            one_hot = self.floatTensor(np.zeros((self.N, self.M)))
            for j in range(self.M):
                one_hot[ind[j], j] = 1
            one_hot = one_hot.view(1, 1, one_hot.size(0), one_hot.size(1))
            one_hot = one_hot.expand(batch_size, 1, one_hot.size(2), one_hot.size(3))
            return one_hot

    # penalty on the duplicated selection
    def regularization(self):
        eps = 1e-10
        z = torch.clamp(torch.softmax(self.qz_loga, dim=0), eps, 1)  # torch.Size([208, 10])
        self.H = torch.sum(F.relu(torch.norm(z, 1, dim=1) - self.thresh))  # L1 norm = sum(abs(xi))
        # print(max(torch.norm(z, 1, dim=1)-self.thresh)) # penalize
        return self.H

    def forward(self, x):
        z = self.sample_z(x.size(0), training=(self.training and not self.freeze))  # torch.Size([16, 1, 44, 3])
        z_t = torch.transpose(z, 2, 3)  # torch.Size([16, 1, 3, 44])
        out = torch.matmul(z_t, x)  # x:torch.Size([16, 1, 44, 1125])
        return out  # out: torch.Size([16, 1, 3, 1125])

def init_weights(m):
    if (type(m) == nn.Linear or type(m) == nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

# net2 = selectionNet(n_chans,5,500,M):  net = selectionNet(n_chans,class_number,wind,channel_to_select) # 81%
class selectionNet(nn.Module):
    def __init__(self,chn_number, class_number, wind_size, M, output_dim=5): #output_dim is the class number
        super(selectionNet, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

        self.N = chn_number
        self.T = wind_size
        self.M = M
        self.output_dim = output_dim

        self.selection_layer = SelectionLayer(self.N, self.M)
        self.network = deepnet(self.M, self.output_dim, self.T)
        #self.network = MSFBCNN(input_dim=[self.M, self.T], output_dim=class_number) # does not converge at all
        #self.network = deepnet(self.M, class_number,input_window_samples=wind_size, final_conv_length='auto')

        self.layers = self.create_layers_field()
        self.apply(init_weights)

    def forward(self, x):
        if isinstance(self.network,MSFBCNN):
            x = torch.unsqueeze(x, dim=1)
            y_selected = self.selection_layer(x)  # x: [16, 1, 44, 1125] y_selected:[16, 1, 3, 1125]
            out = self.network(y_selected)
        elif isinstance(self.network,deepnet):
            x=torch.unsqueeze(x,dim=1) # torch.Size([10, 208, 500])
            y_selected = self.selection_layer(x) #torch.Size([10, 1, 10, 500])
            y_selected = torch.squeeze(y_selected) #torch.Size([10, 10, 500])
            #y_selected.permute(0,2,1)
            if len(y_selected.shape)==2:
                y_selected=y_selected[:,None,:]
            out = self.network(y_selected)
        return out

    def regularizer(self, lamba, weight_decay):
        # Regularization of selection layer
        reg_selection = self.floatTensor([0])
        # L2-Regularization of other layers
        reg = self.floatTensor([0])
        for i, layer in enumerate(self.layers): # selection subnet(one selectionLayer) and deepconv (multiple Conv2D+linear)
            if (type(layer) == SelectionLayer):
                reg_selection += layer.regularization()
            else:
                reg += torch.sum(torch.pow(layer.weight, 2))
        reg = weight_decay * reg + lamba * reg_selection
        return reg

    def create_layers_field(self):
        layers = []
        for idx, m in enumerate(self.modules()):
            if (type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == SelectionLayer):
                layers.append(m)
        return layers

    def get_num_params(self):
        t = 0
        for i, layer in enumerate(self.layers):
            print('Layer ' + str(i))
            print(layer)
            n = 0
            for p in layer.parameters():
                n += np.prod(np.array(p.size()))
            print('Amount of parameters:' + str(n))
            t += n
        print('Total amount of parameters ' + str(t))
        return t

    def set_temperature(self, temp):
        m = self.selection_layer
        m.temperature = temp

    def set_thresh(self, thresh):
        m = self.selection_layer
        m.thresh = thresh

    def monitor(self):
        m = self.selection_layer
        eps = 1e-10
        # Probability distributions
        z = torch.clamp(torch.softmax(m.qz_loga, dim=0), eps, 1)
        # Normalized entropy
        H = - torch.sum(z * torch.log(z), dim=0) / math.log(self.N)
        # Selections
        s = torch.argmax(m.qz_loga, dim=0) + 1
        return H, s, z

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def set_freeze(self, x):
        # freeze selection layer
        m = self.selection_layer
        if (x):
            for param in m.parameters():
                param.requires_grad = False
            m.freeze = True
        else:
            for param in m.parameters():
                param.requires_grad = True
            m.freeze = False

