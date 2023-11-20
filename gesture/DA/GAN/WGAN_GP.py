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

from common_dl import Tensor,device
from gesture.DA.GAN.models import deepnet
from gesture.DA.VAE.VAE import UnFlatten_ch_num, UnFlatten, SEEG_CNN_VAE
from gesture.models.d2l_resnet import d2lresnet
from gesture.utils import read_sids, read_channel_number

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## to work on CPU
#CPU_only=True
#if CPU_only==True:
#    device = torch.device('cpu')
#    Tensor = torch.FloatTensor

learning_rate = 0.0001 # 0.0001/2
weight_decay = 0.0001
b1=0.5 # default=0.9; decay of first order momentum of gradient
b2=0.999
dropout_level = 0.05
latent_dims = 512  # 128/256/1024

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class SEEG_CNN_Generator(nn.Module):
    def __init__(self,sid,chn_num,norm_method):
        super(SEEG_CNN_Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dims, 640),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=22, stride=4),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=18, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=16, stride=4),
            nn.AvgPool1d(10,stride=3,padding=4),#nn.Linear(1500, 500),
            nn.PReLU() # nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = out.view(out.size(0), 16, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
latent_dims=1024
class SEEG_CNN_Generator5(nn.Module):
    def __init__(self,chn_num):
        super(SEEG_CNN_Generator5, self).__init__()
        self.chn_num=chn_num
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dims, self.chn_num*40),
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
            nn.AvgPool1d(10,stride=3,padding=4),#nn.Linear(1500, 500),
            nn.PReLU() # nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = out.view(out.size(0), self.chn_num, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Discriminator(nn.Module):
    def __init__(self,chn_num):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=chn_num, out_channels=16, kernel_size=10, stride = 2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(
            nn.Linear(1968, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

class SEEG_CNN_Generator4(nn.Module):
    def __init__(self,sid,norm_method):
        super(SEEG_CNN_Generator4, self).__init__()

        self.latent_dims = latent_dims
        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dims, 640),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=30, stride=2),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=30, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=14, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layer1(z) # torch.Size([5, 64])-->torch.Size([5, 640])
        out = out.view(out.size(0), 16, 40) # torch.Size([5, 16, 40])
        out = self.layer2(out) # torch.Size([5, 16, 178])
        out = self.layer3(out) # torch.Size([5, 16, 372])
        out = self.layer4(out) # torch.Size([5, 2, 1500])
        return out

class SEEG_CNN_Generator3(nn.Module):
    def __init__(self,sid,norm_method):
        super(SEEG_CNN_Generator3, self).__init__()

        self.latent_dims = latent_dims
        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dims, 640),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=30, stride=2),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=30, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=14, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layer1(z) # torch.Size([5, 64])-->torch.Size([5, 640])
        out = out.view(out.size(0), 16, 40) # torch.Size([5, 16, 40])
        out = self.layer2(out) # torch.Size([5, 16, 178])
        out = self.layer3(out) # torch.Size([5, 16, 372])
        out = self.layer4(out) # torch.Size([5, 2, 1500])
        return out

class G_addition(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,act_function):
        super(G_addition, self).__init__()
        self.addition = nn.Sequential(
        nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride) # torch.Size([32, 16, 82])
        ,act_function) # ,nn.LeakyReLU()
    def forward(self,x):
        y=self.addition(x)
        return y


class SEEG_CNN_Generator2(nn.Module):
    def __init__(self,sid,norm_method):
        sid=10
        super(SEEG_CNN_Generator2, self).__init__()
        self.norm_method=norm_method
        self.fc = nn.Linear(latent_dims, 640)  # torch.Size([32, 640])
        self.uf=UnFlatten(channel_number=8)  # torch.Size([32, UnFlatten_ch_num, -1]): torch.Size([32, 8, 80])
        self.reshape=nn.Sequential(self.fc,self.uf)
        self.act=nn.PReLU() # nn.LeakyReLU()

        self.conv8_16 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=20, stride=1),  # torch.Size([32, 16, 82])
            self.act )
        self.conv16_32 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=20, stride=1),  # torch.Size([32, 16, 82])
            self.act )
        self.conv32_64= nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=20, stride=1),  # torch.Size([32, 16, 82])
            self.act)
        self.conv64_128 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=20, stride=1),  # torch.Size([32, 16, 82])
            self.act)
        self.conv128_256 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=20, stride=1),  # torch.Size([32, 16, 82])
            self.act)
        self.tconv_seq=[self.conv8_16,self.conv16_32,self.conv32_64,self.conv64_128,self.conv128_256]
        x=torch.randn(32,8,80)
        if sid == 2: # 115
            self.seq = nn.Sequential( *self.tconv_seq[:3]) # torch.Size([32, 64, 137])
            self.addition = G_addition(64, 115, 20, 4,self.act)  # torch.Size([32, 115, 564])
            #self.addition=G_addition(64,115,92,3) # torch.Size([32, 115, 500])
        elif sid ==3: #179
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 179, 20, 4,self.act)  # torch.Size([32, 179, 640])
        elif sid == 4:  # 60
            self.seq = nn.Sequential(*self.tconv_seq[:2]) # torch.Size([32, 32, 118])
            self.addition = G_addition(32, 60, 20, 4,self.act)  # torch.Size([32, 60, 488])
        elif sid == 5:  # 178
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 178, 20, 4,self.act)  # torch.Size([32, 178, 640])
        elif sid == 7:  # 143
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 143, 20, 4,self.act)  # torch.Size([32, 143, 640])
        elif sid == 8:  # 168
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 168, 20, 4,self.act)  # torch.Size([32, 168, 640])
        elif sid == 9:  # 114
            self.seq = nn.Sequential(*self.tconv_seq[:3]) # torch.Size([32, 64, 137])
            self.addition = G_addition(64, 114, 20, 4,self.act)  # torch.Size([32, 114, 564])
        elif sid == 10:  # 208
            self.seq = nn.Sequential(*self.tconv_seq[:4]) #torch.Size([32, 128, 156])
            #self.addition = G_addition(128, 208, 20, 4)  # torch.Size([32, 208, 640])
            self.addition = G_addition(128, 208, 35, 3,self.act) # torch.Size([32, 208, 500])
        elif sid == 102:  # 208
            self.seq = nn.Sequential(*self.tconv_seq[:4]) #torch.Size([32, 128, 156])
            #self.addition = G_addition(128, 208, 20, 4)  # torch.Size([32, 208, 640])
            self.addition = G_addition(128, 2, 35, 3,self.act) # torch.Size([32, 208, 500])
        elif sid == 11:  # 191
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 191, 20, 4,self.act)  # torch.Size([32, 191, 640])
        elif sid == 12:  # 92
            self.seq = nn.Sequential(*self.tconv_seq[:3]) # torch.Size([32, 64, 137])
            self.addition = G_addition(64, 92, 20, 4,self.act)  # torch.Size([32, 92, 564])
        elif sid == 13:  # 102
            self.seq = nn.Sequential(*self.tconv_seq[:3]) # torch.Size([32, 64, 137])
            self.addition = G_addition(64, 102, 20, 4,self.act)  # torch.Size([32, 102, 564])
        elif sid == 14:  # 130
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 130, 20, 4,self.act)  # torch.Size([32, 130, 640])
        elif sid == 16:  # 170
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 170, 20, 4,self.act)
        elif sid == 17:  # 139
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 139, 20, 4,self.act)
        elif sid == 18:  # 143
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 143, 20, 4,self.act)
        elif sid == 19:  # 129
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 129, 20, 4,self.act)
        elif sid == 20:  # 107
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 107, 20, 4,self.act)
        elif sid == 21:  # 114
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 114, 20, 4,self.act)
        elif sid == 22:  # 148
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 148, 20, 4,self.act)
        elif sid == 23:  # 195
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 195, 20, 4,self.act)
        elif sid == 24:  # 128
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 128, 20, 4,self.act)
        elif sid == 25:  # 136
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 136, 20, 4,self.act)
        elif sid == 26:  # 153
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 153, 20, 4,self.act)
        elif sid == 29:  # 110
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 110, 20, 4,self.act)
        elif sid == 30:  # 107
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 107, 20, 4,self.act)
        elif sid == 31:  # 71
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 71, 20, 4,self.act)
        elif sid == 32:  # 56
            self.seq = nn.Sequential(*self.tconv_seq[:2])
            self.addition = G_addition(32, 56, 20, 4,self.act)
        elif sid == 34:  # 101
            self.seq = nn.Sequential(*self.tconv_seq[:3])
            self.addition = G_addition(64, 101, 20, 4,self.act)
        elif sid == 35:  # 136
            self.seq = nn.Sequential(*self.tconv_seq[:4])
            self.addition = G_addition(128, 136, 20, 4,self.act)
        elif sid == 41:  # 190
            self.seq = nn.Sequential(*self.tconv_seq[:4]) # torch.Size([32, 128, 156])
            self.addition = G_addition(128, 190, 20, 4,self.act) # torch.Size([32, 190, 640])
            #self.addition = G_addition(128, 190, 35, 3)
        self.format=nn.Sequential(self.seq,self.addition)
        test_out=self.format(x)

        ## the Linear layer will cause training unstable. The format layer should ouput the final correct shape.
        #self.last=nn.Sequential(nn.Linear(in_features=test_out.shape[2],out_features=500),nn.Sigmoid())
        self.minmax = nn.Sequential(nn.Sigmoid(),) # for loading original data using the min-max normalization
    def forward(self,z):
        #recon = self.decode(z)
        z = self.reshape(z) # torch.Size([bs, 8, 80])
        z = self.format(z)
        if self.norm_method=='minmax':
            z = self.minmax(z)
        return z

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

class_number=5
wind=500
lambda_gp=10
critic='deepnet' # resnet/deepnet
def wgan_gp(writer,sid,continuous, chn_num, class_num, wind_size, scaler,norm_method,train_loader, epochs,save_gen_dir, result_dir, classi,total_trials):
    class_num=1 # in GAN, there are only two class, and a scalar value should be produced
    #generator = SEEG_CNN_Generator(sid,chn_num,norm_method).to(device) # good
    generator = SEEG_CNN_Generator5(chn_num).to(device)  # SEEG_CNN_Generator().to(device)
    if critic=='deepnet':
        discriminator = deepnet(chn_num,class_num,wind_size).to(device) # SEEG_CNN_Discriminator().to(device)
    elif critic=='resnet':
        discriminator = d2lresnet(class_num,as_DA_discriminator=True).to(device)
    discriminator = Discriminator(chn_num).to(device)
    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,  weight_decay=weight_decay, betas=(b1, b2))

    if continuous=='resume':
        #### load model from folder generated by tensorboard #####
        path = Path(writer.log_dir)
        parient_dir=path.parent.absolute()
        folder_list = realsorted([str(pth) for pth in parient_dir.iterdir()])# if pth.suffix == '.npy'])
        pth_folder=folder_list[-2] # previous folder
        pth_list = realsorted([str(pth) for pth in Path(pth_folder).iterdir() if pth.name.startswith('checkpoint_D_epochs_')])
        pth_file_D=pth_list[-1]
        pth_list = realsorted([str(pth) for pth in Path(pth_folder).iterdir() if pth.name.startswith('checkpoint_G_epochs_')])
        pth_file_G = pth_list[-1]

        #### load model from folder outside of tensorboard #####
        #pth_file_D = result_dir + 'checkpoint_D_epochs_*'+ '.pth'
        #pth_file_D = os.path.normpath(glob.glob(pth_file_D)[-1]) # -1 is the largest epoch number
        #pth_file_G = result_dir + 'checkpoint_G_epochs_*' + '.pth'
        #pth_file_G = os.path.normpath(glob.glob(pth_file_G)[-1])

        checkpoint_D = torch.load(pth_file_D)
        checkpoint_G = torch.load(pth_file_G)
        discriminator.load_state_dict(checkpoint_D['net'])
        generator.load_state_dict(checkpoint_G['net'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        #### load model #####

        pre_epochs=checkpoint_D['epoch']
        print('Resume training. The last training epoch is '+str(pre_epochs)+'.')

    elif continuous=='fresh':
        pre_epochs=0
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    else:
        print("Python: Mush be either 'resume' or 'fresh'.")
        return -1

    global_steps=(pre_epochs+1) * len(train_loader)

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print (interpolates.shape)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    d_losses=[]
    g_losses=[]
    d_loss_item=0
    g_loss_item=0

    plot_fig_number = 4
    #plot_channel_num=min(5,chn_num)
    plot_channel_num=2
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    fig.tight_layout()

    for epoch in range(pre_epochs,pre_epochs+epochs):
        discriminator.train()
        generator.train()
        for i, data in enumerate(train_loader):
            x, _ = data
            x = x.to(device)
            real_data = x.type(Tensor) # torch.Size([5, 2, 1500])

            if i % 1 == 0: # 0
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # train with real
                optimizer_D.zero_grad()
                output_real = discriminator(real_data) # scalar value torch.Size([32, 1])

                # train with fake
                # Generate fake data
                z = torch.randn(x.shape[0], latent_dims).to(device)  # torch.Size([32, 128])
                fake_data = generator(z) # torch.Size([32, 208, 500])
                output_fake = discriminator(fake_data) # torch.Size([32, 1])
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)
                d_loss = -torch.mean(output_real) + torch.mean(output_fake) + lambda_gp * gradient_penalty
                d_loss_item=d_loss.item()
                d_loss.backward()
                optimizer_D.step()
                writer.add_scalar('d_loss', d_loss_item, global_steps)
            # Train the generator every n_critic steps
            if ((continuous=='fresh' and epoch > 0) or (continuous=='resume')) and i % 5 == 0:
            #if 1 == 1:
                # print('Generator')
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                z = torch.randn(x.shape[0], latent_dims).to(device) # torch.Size([32, 1024])
                fake_data = generator(z) # torch.Size([32, 208, 500])
                output_fake = discriminator(fake_data) # torch.Size([32, 1])
                g_loss = -torch.mean(output_fake)
                g_loss_item=g_loss.item()
                g_loss.backward()
                optimizer_G.step()
                writer.add_scalar('g_loss', g_loss_item, global_steps)
            d_losses.append(d_loss_item)
            g_losses.append(g_loss_item)
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
            epoch, pre_epochs+epochs, i, len(train_loader), d_loss_item, g_loss_item))
            global_steps+=1

        if (epoch + 1) % 10 == 0:
            #def gen_data_wgangp(sid, chn_num, class_num, wind_size, result_dir, num_data_to_generate=500):
            num_data_to_generate=5
            gen_data_tmp = gen_data_wgangp_(generator, latent_dims, num_data_to_generate=plot_fig_number) # (batch_size, 208, 500)
            ''' generate in another way
            trials,labels=next(iter(test_loader))
            trial,label=trials[0],labels[0] # torch.Size([208, 500])
            trial=torch.unsqueeze(trial,0)
            if generative_model=='VAE':
                gen_trial, mu, logvar=model(trial.type(Tensor))
            trial, gen_trial=torch.squeeze(trial), torch.squeeze(gen_trial)
            '''
            #### scaling back ####
            scaling_back=False
            if scaling_back==True:
                gen_data = np.zeros((gen_data_tmp.shape))
                for i, trial in enumerate(gen_data_tmp):
                    tmp = scaler.inverse_transform(trial.transpose())
                    gen_data[i] = np.transpose(tmp)
            else:
                gen_data=gen_data_tmp
            #### scaling back ####

            plot_buf = gen_plot(axs, gen_data,plot_channel_num)
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            # writer = SummaryWriter(comment='synthetic signals')
            writer.add_image('Image', image[0], global_steps)

            # trial, gen_trial=trial.transpose(), gen_trial.transpose()
            # compare original vs generated data
            save_gen_data=False
            if save_gen_data==True:
                print("Saving generated data of class " + str(classi) + ".")
                if classi == 0:
                    np.save(save_gen_dir + 'gen_class_0_'+str(epoch+1)+'.npy', gen_data)
                elif classi == 1:
                    np.save(save_gen_dir + 'gen_class_1_'+str(epoch+1)+'.npy', gen_data)
                elif classi == 2:
                    np.save(save_gen_dir + 'gen_class_2_'+str(epoch+1)+'.npy', gen_data)
                elif classi == 3:
                    np.save(save_gen_dir + 'gen_class_3_'+str(epoch+1)+'.npy', gen_data)
                elif classi == 4:
                    np.save(save_gen_dir + 'gen_class_4_'+str(epoch+1)+'.npy', gen_data)
            state_D = {
                'net': discriminator.state_dict(),
                'optimizer': optimizer_D.state_dict(),
                'epoch': epoch+pre_epochs,
                # 'loss': epoch_loss
            }
            state_G = {
                'net': generator.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'epoch': epoch+pre_epochs,
                # 'loss': epoch_loss
            }
            #if continuous==True:
            #    savepath_D = result_dir + 'checkpoint_D_continuous_' + str(epoch) + '.pth'
            #    savepath_G = result_dir + 'checkpoint_G_continuous_' + str(epoch) + '.pth'
            #else:
            savepath_D = writer.log_dir + 'checkpoint_D_epochs_' + str(epoch+pre_epochs) + '.pth'
            savepath_G = writer.log_dir + 'checkpoint_G_epochs_' + str(epoch+pre_epochs) + '.pth'
            torch.save(state_D, savepath_D)
            torch.save(state_G, savepath_G)

    losses = {'d_losses': d_losses,
              'g_losses': g_losses}
    filename = result_dir + 'losses_class_' + str(classi) + '_' + critic + '_lat_' + str(
        latent_dims) + '_epochs_' + str(epochs + pre_epochs) + '.npy'
    np.save(filename, losses) # use tensorboard

    return

# Generating new data
#gen_data_tmp = gen_data_gan(sid, chn_num, class_num, wind_size, latent_dims,result_dir, num_data_to_generate=5)
def gen_data_wgangp(sid,chn_num, class_num, wind_size,latent_dims,result_dir, num_data_to_generate=500):
    generator = SEEG_CNN_Generator2(sid).to(device)
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
    #optimizer_D.load_state_dict(checkpoint_D['optimizer'])
    #optimizer_G.load_state_dict(checkpoint_G['optimizer'])
    gen_data=gen_data_wgangp_(generator,latent_dims,num_data_to_generate)
    return gen_data

def gen_data_wgangp_(generator,latent_dims,num_data_to_generate=111):
    new_data = []
    with torch.no_grad():
        generator.eval()
        for epoch in range(num_data_to_generate):
            z = torch.randn(1, latent_dims).to(device)
            gen_data = generator(z) # torch.Size([1, 208, 500])
            new_data.append(gen_data[0, :, :].cpu().numpy()) # new_data[0]: torch.Size([208, 500])
        new_data = np.asarray(new_data) # (500, 208, 500)
        return new_data

def gen_plot(axs,gen_data,plot_channel_num):
    for i in range(2): # 4 plots
        for j in range(2):
            axs[i,j].clear()
            for k in range(plot_channel_num):
                axs[i,j].plot(gen_data[i*2+j,k,:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf



