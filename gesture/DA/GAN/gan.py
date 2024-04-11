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
            nn.AvgPool1d(10, stride=3, padding=4),  # nn.Linear(1500, 500),
            nn.PReLU()  # nn.Sigmoid()
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

def gan(method,writer, sid, continuous, chn_num, class_num, wind_size, train_loader, epochs,latent_dims):
    #batch_size=train_loader.batch_size  # last batch is not the same
    class_num = 1  # in GAN, there are only two class, and a scalar value should be produced
    generator = SEEG_CNN_Generator_10channels(chn_num,latent_dims).to(device)  # SEEG_CNN_Generator().to(device)
    discriminator = deepnet(method,chn_num, class_num, wind_size).to(device)  # SEEG_CNN_Discriminator().to(device)
    #discriminator = Discriminator(chn_num).to(device)
    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                   betas=(b1, b2))

    if continuous == 'resume':
        #### load model from folder generated by tensorboard #####
        path = Path(writer.log_dir)
        parient_dir = path.parent.absolute()
        folder_list = realsorted([str(pth) for pth in parient_dir.iterdir()])  # if pth.suffix == '.npy'])
        pth_folder = folder_list[-2]  # previous folder
        pth_list = realsorted(
            [str(pth) for pth in Path(pth_folder).iterdir() if pth.name.startswith('checkpoint_D_epochs_')])
        pth_file_D = pth_list[-1]
        pth_list = realsorted(
            [str(pth) for pth in Path(pth_folder).iterdir() if pth.name.startswith('checkpoint_G_epochs_')])
        pth_file_G = pth_list[-1]

        #### load model from folder outside of tensorboard #####
        # pth_file_D = result_dir + 'checkpoint_D_epochs_*'+ '.pth'
        # pth_file_D = os.path.normpath(glob.glob(pth_file_D)[-1]) # -1 is the largest epoch number
        # pth_file_G = result_dir + 'checkpoint_G_epochs_*' + '.pth'
        # pth_file_G = os.path.normpath(glob.glob(pth_file_G)[-1])

        checkpoint_D = torch.load(pth_file_D)
        checkpoint_G = torch.load(pth_file_G)
        discriminator.load_state_dict(checkpoint_D['net'])
        generator.load_state_dict(checkpoint_G['net'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        #### load model #####

        pre_epochs = checkpoint_D['epoch']
        global_steps = (pre_epochs + 1) * len(train_loader)
        print('Resume training. The last training epoch is ' + str(pre_epochs) + '.')

    elif continuous == 'fresh':
        pre_epochs = 0
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        global_steps = 0
    else:
        print("Python: Mush be either 'resume' or 'fresh'.")
        return -1

    d_losses = []
    g_losses = []
    d_loss_item = 0
    g_loss_item = 0

    plot_fig_number = 4
    # plot_channel_num=min(5,chn_num)
    plot_channel_num = 2
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    fig.tight_layout()

    for epoch in range(pre_epochs, pre_epochs + epochs):
        discriminator.train()
        generator.train()
        for i, data in enumerate(train_loader):
            # real
            x, real_label = data
            real_data,real_label = x.type(Tensor).to(device), real_label.type(LongTensor).to(device)
            batch_size=real_data.shape[0]
            #### train discriminator ####
            if i % 1 == 0:  # 0
                optimizer_D.zero_grad()

                #### forward pass ####
                # fake
                z = torch.randn(x.shape[0], latent_dims).to(device)  # torch.Size([32, 128])
                if method == 'CWGANGP':
                    fake_label = real_label  # , for GP calculation
                elif method == 'CDCGAN':
                    fake_label = torch.randint(0, 5, real_label.shape).to(device)
                fake_data = generator(z, fake_label)  # torch.Size([32, 208, 500])

                # TODO: test using 0.9 and 0.1 for valid and fake label
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
                real_validity = discriminator(real_data, real_label)  # scalar value torch.Size([32, 1])
                fake_validity = discriminator(fake_data, fake_label)  # torch.Size([32, 1])
                ####  forward pass ####

                if method == 'CWGANGP':
                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data,fake_label)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                    d_loss_item = d_loss.item()
                    d_loss.backward(retain_graph=False)#d_loss.backward(retain_graph=True) # multiple backward propagation(error prone)
                    optimizer_D.step()
                    writer.add_scalar('d_loss', d_loss_item, global_steps)
                elif method=='CDCGAN':
                    # Calculate error and backpropagate
                    d_loss_real = adversarial_loss(real_validity, valid) # real_label
                    d_loss_fake = adversarial_loss(fake_validity, fake) # fake_label
                    d_loss=(d_loss_real+d_loss_fake)/2 # ? /2 ?
                    d_loss.backward(retain_graph=False)
                    optimizer_D.step()
                    d_loss_item = d_loss.item()

                    # monitoring
                    r_correct_adv = sum(torch.sigmoid(real_validity) > 0.5)  # sum(torch.sum(r_preds_adv.squeeze() == r_label_adv.squeeze())
                    f_correct_adv = sum(torch.sigmoid(fake_validity) < 0.5)  # torch.sum(f_preds_adv.squeeze() == f_label_adv.squeeze())
                    r_acc_adv = r_correct_adv / batch_size
                    f_acc_adv = f_correct_adv / batch_size
                    writer.add_scalars('monitor', {'r_acc_adv': r_acc_adv,
                                                   'f_acc_adv': f_acc_adv,},
                                       global_steps)

            #### train generator ####
            if i % 1 == 0: # (continuous == 'fresh' and epoch > 0) or (continuous == 'resume'):
                optimizer_G.zero_grad()

                #### forward pass ####
                # fake
                z = torch.randn(x.shape[0], latent_dims).to(device)  # torch.Size([32, 128])
                if method == 'CWGANGP':
                    fake_label = real_label  # , for GP calculation
                elif method == 'CDCGAN':
                    fake_label = torch.randint(0, 5, real_label.shape).to(device)
                fake_data = generator(z, fake_label)  # torch.Size([32, 208, 500])

                # TODO: test using 0.9 and 0.1 for valid and fake label
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
                real_validity = discriminator(real_data, real_label)  # scalar value torch.Size([32, 1])
                fake_validity = discriminator(fake_data, fake_label)  # torch.Size([32, 1])
                ####  forward pass ####

                if method=='CWGANGP':
                    g_loss = -torch.mean(fake_validity)
                    g_loss_item = g_loss.item()
                    g_loss.backward()
                    optimizer_G.step()
                    writer.add_scalar('g_loss', g_loss_item, global_steps)
                elif method == 'CDCGAN':
                    g_loss = adversarial_loss(real_validity, valid)
                    g_loss_item = g_loss.item()
                    g_loss.backward()
                    optimizer_G.step()
                    writer.add_scalar('g_loss', g_loss_item, global_steps)

            d_losses.append(d_loss_item)
            g_losses.append(g_loss_item)
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
                epoch, pre_epochs + epochs, i, len(train_loader), d_loss_item, g_loss_item))
            global_steps += 1

        if (epoch + 1) % 1 == 0:
            # def gen_data_wgangp(sid, chn_num, class_num, wind_size, result_dir, num_data_to_generate=500):
            gen_data = gen_data_wgangp_(generator, latent_dims,num_data_to_generate=plot_fig_number)  # (batch_size, 208, 500)

            plot_buf = gen_plot(axs, gen_data, plot_channel_num)
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            # writer = SummaryWriter(comment='synthetic signals')
            writer.add_image('Image', image[0], global_steps)

            state_D = {
                'net': discriminator.state_dict(),
                'optimizer': optimizer_D.state_dict(),
                'epoch': epoch + pre_epochs,
                # 'loss': epoch_loss
            }
            state_G = {
                'net': generator.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'epoch': epoch + pre_epochs,
                # 'loss': epoch_loss
            }
            # if continuous==True:
            #    savepath_D = result_dir + 'checkpoint_D_continuous_' + str(epoch) + '.pth'
            #    savepath_G = result_dir + 'checkpoint_G_continuous_' + str(epoch) + '.pth'
            # else:
            if epoch>400:
                savepath_D = writer.log_dir + 'checkpoint_D_epochs_' + str(epoch + pre_epochs) + '.pth'
                savepath_G = writer.log_dir + 'checkpoint_G_epochs_' + str(epoch + pre_epochs) + '.pth'
                torch.save(state_D, savepath_D)
                torch.save(state_G, savepath_G)

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



