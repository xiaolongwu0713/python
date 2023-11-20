import torch
import torch.nn as nn
import numpy as np
import os,io
import random
import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

from common_dl import Tensor,device
from gesture.models.deepmodel import deepnet
from gesture.DA.VAE.VAE import UnFlatten_ch_num, UnFlatten, SEEG_CNN_VAE
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


batch_size = 5
learning_rate = 0.0001
b1=0.5 # decay of first order momentum of gradient
b2=0.99
dropout_level = 0.05
latent_dims = 64


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class SEEG_CNN_Generator(nn.Module):
    def __init__(self):
        super(SEEG_CNN_Generator, self).__init__()

        self.latent_dims = latent_dims
        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dims, 640),
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
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layer1(z) # torch.Size([5, 64])-->torch.Size([5, 640])
        out = out.view(out.size(0), 16, 40) # torch.Size([5, 16, 40])
        out = self.layer2(out) # torch.Size([5, 16, 178])
        out = self.layer3(out) # torch.Size([5, 16, 372])
        out = self.layer4(out) # torch.Size([5, 2, 1500])
        return out

class SEEG_CNN_Generator2(nn.Module):
    def __init__(self,channel_number,latent_dims):
        super(SEEG_CNN_Generator2,self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(int(channel_number*latent_dims/2), channel_number*latent_dims),nn.PReLU(),UnFlatten(channel_number))  # torch.Size([32, 640])
        self.layer2=nn.Sequential(
            nn.ConvTranspose1d(in_channels=channel_number, out_channels=channel_number, kernel_size=10, stride=3),  # torch.Size([32, 16, 82])
            nn.PReLU(),)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channel_number, out_channels=channel_number, kernel_size=48, stride=1),  # torch.Size([32, 64, 182])
            nn.PReLU(),)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channel_number, out_channels=channel_number, kernel_size=10, stride=2),  # torch.Size([32, 128, 382])
            nn.Sigmoid())
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channel_number, out_channels=channel_number, kernel_size=51, stride=1),  # torch.Size([32, 208, 782])
            nn.Sigmoid()  # between [0,1]
        )

    def forward(self,z):
        output = self.layer1(z)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        #output = self.layer5(output)

        return output

class EEG_CNN_Discriminator2(nn.Module):
    def __init__(self,channel_number):
        super(EEG_CNN_Discriminator2,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=channel_number, out_channels=channel_number, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=channel_number),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)) #torch.Size([32, 208, 123])
        self.dense_layers = nn.Sequential(
            nn.Linear(25584, 10000),
            nn.LeakyReLU(0.2),
            nn.Linear(10000, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

class SEEG_CNN_Discriminator(nn.Module):
    def __init__(self):
        super(SEEG_CNN_Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(
            nn.Linear(5968, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

n_chans=208
class_number=5
wind=500

def dcgan(writer,sid,chn_num, class_num, wind_size, scaler,train_loader, num_epochs,save_gen_dir,result_dir,classi):
    class_num=1 # in GAN, there are only two class, and a scalar value should be produced
    generator = SEEG_CNN_Generator2(chn_num,latent_dims).to(device) # SEEG_CNN_Generator().to(device)
    #discriminator = deepnet(chn_num,class_num,wind_size).to(device) # SEEG_CNN_Discriminator().to(device)
    discriminator = EEG_CNN_Discriminator2(chn_num).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

    d_losses = []
    g_losses = []
    d_loss_item = 0
    g_loss_item = 0
    plot_fig_number = 4
    plot_channel_num = 5
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    fig.tight_layout()

    # DA Training ---------------------------------------------------------------

    for epoch in range(num_epochs):
        discriminator.train()
        generator.train()
        for i, data in enumerate(train_loader, 0):
            global_steps=epoch*len(train_loader)+i
            x, _ = data
            x = x.to(device)

            # Configure input
            real_data = x.type(Tensor) # torch.Size([5, 2, 1500])
            label = torch.ones(x.shape[0],1).to(device)
            z = torch.randn(x.shape[0], int(latent_dims*chn_num/2)).to(device) #torch.Size([32, 128])

            if i % 5 == 0: # 5
                # print('Discriminator')
                # ---------------------
                #  Train Discriminator
                # ---------------------

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # train with real
                optimizer_Dis.zero_grad()
                output_real = discriminator(real_data) # scalar value torch.Size([32, 1])

                # Calculate error and backpropagate
                errD_real = adversarial_loss(output_real, label)
                errD_real.backward()

                # train with fake
                # Generate fake data
                fake_data = generator(z) # torch.Size([32, 208, 500])
                label = torch.zeros(x.shape[0], 1).to(device)
                output_fake = discriminator(fake_data) # torch.Size([32, 1])
                errD_fake = adversarial_loss(output_fake, label)
                errD_fake.backward()
                optimizer_Dis.step()

                errD = errD_real + errD_fake
                d_losses.append(errD.item())
                writer.add_scalar('d_loss', errD.item(), global_steps)

            # Train the generator every n_critic steps
            if i % 1 == 0:
                # print('Generator')
                # -----------------
                #  Train Generator
                # -----------------

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                z = torch.randn(x.shape[0], int(latent_dims*chn_num/2)).to(device) # torch.Size([32, 128])
                fake_data = generator(z) # torch.Size([32, 208, 500])
                # Reset gradients
                optimizer_Gen.zero_grad()

                output = discriminator(fake_data) # torch.Size([32, 1])
                ## ones??? not zeros??
                bceloss = adversarial_loss(output, torch.ones(x.shape[0], 1).to(device))
                errG = bceloss
                errG.backward()
                optimizer_Gen.step()

                g_losses.append(errG.item())
                writer.add_scalar('g_loss', errG.item(), global_steps)

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
            epoch, num_epochs, i, len(train_loader), errD.item(), errG.item(),))

        if epoch%10==0:
            gen_data_tmp = gen_data_gan(generator,discriminator,train_loader.batch_size, chn_num,plot_fig_number)
            plot_buf = gen_plot(axs, gen_data_tmp, plot_fig_number, plot_channel_num)
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            # writer = SummaryWriter(comment='synthetic signals')
            writer.add_image('Image', image[0], global_steps)
    losses = {'d_losses': d_losses,
              'g_losses': g_losses}
    filename = result_dir + 'losses_class_' + str(classi) + '_lat_' + str(
        latent_dims) + '_epochs_' + str(epochs + pre_epochs) + '.npy'
    np.save(filename, losses)  # use tensorboard

    gen_data_tmp = gen_data_gan(generator,discriminator, 500)  # (500, 208, 500)
    ''' generate in another way
    trials,labels=next(iter(test_loader))
    trial,label=trials[0],labels[0] # torch.Size([208, 500])
    trial=torch.unsqueeze(trial,0)
    if generative_model=='VAE':
        gen_trial, mu, logvar=model(trial.type(Tensor))
    trial, gen_trial=torch.squeeze(trial), torch.squeeze(gen_trial)
    '''
    ## scaling back
    scalling_back=False
    if scalling_back:
        gen_data = np.zeros((gen_data_tmp.shape))
        for i, trial in enumerate(gen_data_tmp):
            tmp = scaler.inverse_transform(trial.transpose())
            gen_data[i] = np.transpose(tmp)
    else:
        gen_data=gen_data_tmp
    # trial, gen_trial=trial.transpose(), gen_trial.transpose()
    # compare original vs generated data
    print("Saving generated data of class " + str(classi) + ".")
    if classi == 0:
        np.save(save_gen_dir + 'gen_class_0.npy', gen_data)
    elif classi == 1:
        np.save(save_gen_dir + 'gen_class_1.npy', gen_data)
    elif classi == 2:
        np.save(save_gen_dir + 'gen_class_2.npy', gen_data)
    elif classi == 3:
        np.save(save_gen_dir + 'gen_class_3.npy', gen_data)
    elif classi == 4:
        np.save(save_gen_dir + 'gen_class_4.npy', gen_data)
    return
    # Generating new data
def gen_data_gan(generator,discriminator, bs, chn_num,num_data_to_generate=500):
    generator.eval()
    discriminator.eval()
    new_data = []
    with torch.no_grad():
        for epoch in range(num_data_to_generate):
            z = torch.randn(bs, int(latent_dims*chn_num/2)).to(device)
            gen_data = generator(z) # torch.Size([1, 208, 500])

            new_data.append(gen_data[0, :, :].cpu().numpy()) # new_data[0]: torch.Size([208, 500])

        new_data = np.asarray(new_data) # (500, 208, 500)
        return new_data

def gen_plot(axs,gen_data,fig_num,plot_channel_num):
    for i in range(2): # 4 plots
        for j in range(2):
            axs[i,j].clear()
            # five channels on each plot
            for k in range(plot_channel_num):
                axs[i,j].plot(gen_data[i*2+j,k,:])

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf