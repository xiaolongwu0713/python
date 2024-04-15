import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from common_dl import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dims = 128
UnFlatten_ch_num=8

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, channel_number):
        super(UnFlatten, self).__init__()
        self.channel_number=channel_number
    def forward(self, input):
        UF = input.view(input.size(0), self.channel_number, -1)
        return UF


class SEEG_CNN_VAE(nn.Module):
    def __init__(self, ch_num, class_num, wind_size):
        super(SEEG_CNN_VAE, self).__init__()
        self.ch_num=ch_num
        self.class_num=class_num
        self.wind_size=wind_size
        self.out_channel=64
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=ch_num, out_channels=self.out_channel, kernel_size=50, stride=2),
            nn.BatchNorm1d(num_features=self.out_channel),
            nn.PReLU(),
            nn.MaxPool1d(10),
            Flatten()
        )
        test=torch.randn(99,self.ch_num,self.wind_size)
        latent_test=self.encoder(test)
        temp_dim=latent_test.shape[1]
        self.fc1 = nn.Linear(temp_dim, latent_dims) # mu
        self.fc2 = nn.Linear(temp_dim, latent_dims) # sigma
        self.fc3 = nn.Linear(latent_dims, 640) # torch.Size([32, 640])

        self.decoder = nn.Sequential(
            UnFlatten(self.ch_num), # b,torch.Size([32, UnFlatten_ch_num, -1]): torch.Size([32, 8, 80])
            nn.ConvTranspose1d(in_channels=self.ch_num, out_channels=self.ch_num, kernel_size=20, stride=1), # torch.Size([32, 16, 82])
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.ch_num, out_channels=self.ch_num, kernel_size=20, stride=2), # torch.Size([32, 64, 182])
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.ch_num, out_channels=self.ch_num, kernel_size=20, stride=2), # torch.Size([32, 128, 382])
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.ch_num, out_channels=self.ch_num, kernel_size=40, stride=1), # torch.Size([32, 208, 782])

        )
        self.final_linear=nn.Sequential(nn.Linear(425,519),nn.Sigmoid(),nn.AvgPool1d(20,1))
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z) # torch.Size([32, 640])z
        z = self.decoder(z) # torch.Size([32, 10, 425])
        z = self.final_linear(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)  #z: torch.Size([32, 128])
        recon = self.decode(z)
        return recon, mu, logvar

## which one to use??
def loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x)
    #recon_loss=((x - recon_x) ** 2).sum()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + KLD, recon_loss, KLD

# Generating new data
def gen_data_vae(model, num_data_to_generate=500):
    print("Generating data.")
    new_data = []
    with torch.no_grad():
        model.eval()
        for epoch in range(num_data_to_generate):
            z = torch.randn(1, 128).to(device)
            recon_data = model.decode(z).detach().cpu().numpy() # shape: (1, 208, 500)
            new_data.append(recon_data[0,:,:])

        new_data = np.asarray(new_data)
        return new_data



















