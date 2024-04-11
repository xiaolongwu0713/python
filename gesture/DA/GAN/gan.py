# This is conditional WGAN without GP;
# There is also a CWGAN with GP example: https://github.com/gcucurull/cond-wgan-gp
import glob
import io

import PIL
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from natsort import realsorted
from torch.autograd import Variable
import torch.autograd as autograd
import os
import random

from torchvision.transforms import ToTensor
from tqdm import tqdm

from common_dl import *
from gesture.DA.GAN.models import deepnet
from gesture.DA.VAE.VAE import UnFlatten_ch_num, UnFlatten, SEEG_CNN_VAE
from gesture.models.d2l_resnet import d2lresnet
from gesture.utils import read_sids, read_channel_number

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## to work on CPU
# CPU_only=True
# if CPU_only==True:
#    device = torch.device('cpu')
#    Tensor = torch.FloatTensor

learning_rate = 0.0001  # 0.0001/2
weight_decay = 0.0001
b1 = 0.5  # default=0.9; decay of first order momentum of gradient
b2 = 0.999
dropout_level = 0.05
#latent_dims = 512  # 128/256/1024


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].reshape(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Discriminator(nn.Module):
    def __init__(self, chn_num):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=chn_num, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(
            nn.Linear(1968, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 1))

    def forward(self, x, labels):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


class SEEG_CNN_Generator_10channels(nn.Module):
    def __init__(self, chn_num,latent_dims):
        super(SEEG_CNN_Generator_10channels, self).__init__()

        self.chn_num = chn_num
        self.label_emb = nn.Embedding(5, self.chn_num) #self.label_emb = nn.Embedding(5, 5)
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dims+self.chn_num, self.chn_num * 40),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.chn_num, out_channels=self.chn_num, kernel_size=22, stride=4),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.chn_num, out_channels=self.chn_num, kernel_size=18, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.chn_num, out_channels=self.chn_num, kernel_size=16, stride=4),
            #nn.AvgPool1d(10, stride=3, padding=4),  # nn.Linear(1500, 500),
            nn.Linear(1500, 500),
            #nn.Sigmoid()
            #nn.PReLU()  # nn.Sigmoid()
        )

    def forward(self, z, label): # emb: torch.Size([32, 1, 10]); z:torch.Size([32, 512])
        z = torch.cat((self.label_emb(label).squeeze(), z), -1) # torch.Size([32, 532])

        out = self.layer1(z)
        out = out.view(out.size(0), self.chn_num, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


# test the dimension: dimensions are good;
'''
sids=read_sids()
n_chans = read_channel_number()
for sid in sids:
    gg=SEEG_CNN_Generator2(sid)
    z = torch.randn(32, latent_dims) # torch.Size([32, 512])
    out=gg(z)
    print(out.shape)
'''

class_number = 5
wind = 500
lambda_gp=10
adversarial_loss = nn.BCEWithLogitsLoss()

def gan(gen_method,writer, sid, continuous, chn_num, class_number, wind, train_loader, epochs,latent_dims):

    return


# Generating new data
# gen_data_tmp = gen_data_gan(sid, chn_num, class_num, wind_size, latent_dims,result_dir, num_data_to_generate=5)
def gen_data_wgangp(sid, chn_num, class_num, wind_size, latent_dims, result_dir, num_data_to_generate=500):
    generator = SEEG_CNN_Generator_10channels(sid).to(device)
    discriminator = deepnet(chn_num, class_num, wind_size).to(device)

    #### load model #####
    savepath_D = result_dir + 'checkpoint_D_epochs_*' + '.pth'
    savepath_D = os.path.normpath(glob.glob(savepath_D)[-1])  # -1 is the largest epoch number
    savepath_G = result_dir + 'checkpoint_G_epochs_*' + '.pth'
    savepath_G = os.path.normpath(glob.glob(savepath_G)[-1])
    checkpoint_D = torch.load(savepath_D)
    checkpoint_G = torch.load(savepath_G)
    discriminator.load_state_dict(checkpoint_D['net'])
    generator.load_state_dict(checkpoint_G['net'])
    # optimizer_D.load_state_dict(checkpoint_D['optimizer'])
    # optimizer_G.load_state_dict(checkpoint_G['optimizer'])
    gen_data = gen_data_wgangp_(generator, latent_dims, num_data_to_generate)
    return gen_data


def gen_data_wgangp_(generator, latent_dims, num_data_to_generate=111):
    class_num=5
    with torch.no_grad():
        generator.eval()
        #for epoch in range(num_data_to_generate):
        z = torch.randn(num_data_to_generate, latent_dims).to(device)
        gen_label = torch.randint(0, class_num, (num_data_to_generate,1)).to(device)
        gen_data = generator(z,gen_label)  # torch.Size([1, 208, 500])
        new_data = gen_data.cpu().numpy()  # (500, 208, 500)
        return new_data


def gen_plot(axs, gen_data, plot_channel_num):
    for i in range(2):  # 4 plots
        for j in range(2):
            axs[i, j].clear()
            for k in range(plot_channel_num):
                axs[i, j].plot(gen_data[i * 2 + j, k, :])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf



