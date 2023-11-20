import PIL
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os,io
import random
import argparse

from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1200, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=320, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
parser.add_argument('--nz', type=int, default=64, help="size of the latent z vector used as the generator input.")
opt = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 5
learning_rate = 0.0001
dropout_level = 0.05

opt.n_epochs=1200
nz = opt.nz

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  


class EEG_CNN_Generator(nn.Module): 
    def __init__(self): 
        super(EEG_CNN_Generator, self).__init__()
        self.nz = nz 
        self.layer1 = nn.Sequential( 
            nn.Linear(self.nz, 640), 
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
        out = self.layer1(z) 
        out = out.view(out.size(0), 16, 40) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        return out 

class EEG_CNN_Discriminator(nn.Module): 
    def __init__(self): 
        super(EEG_CNN_Discriminator, self).__init__() 
 
        self.layer1 = nn.Sequential( 
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride = 2), 
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

def gen_data(generator,discriminator,dataloader,Tensor,plot_fig_number):
    discriminator.eval()
    generator.eval()
    new_data=[]
    with torch.no_grad():
        for epoch in range(plot_fig_number):
            z = torch.randn(dataloader.batch_size, nz).to(device)
            fake_data = generator(z) # torch.Size([5, 2, 1500])
            fake_data = fake_data.data[0].cpu().numpy() # (5, 2, 1500)
            new_data.append(fake_data)
    new_data=np.asarray(new_data).squeeze()
    return new_data

def gen_plot(axs,gen_data,plot_channel_num):
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

def dcgan(writer,datatrain, label, nseed):
    plot_fig_number = 4
    plot_channel_num = 2

    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    generator = EEG_CNN_Generator().to(device)
    discriminator = EEG_CNN_Discriminator().to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(opt.b1, opt.b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    real_label = 1
    fake_label = 0
    batches_done = 0
    new_data = []
    global_step = 0

    # GAN Training ---------------------------------------------------------------
    plot_fig_number = 4
    plot_channel_num = 2
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    fig.tight_layout()
    for epoch in range(opt.n_epochs):
        discriminator.train()
        generator.train()
        for i, data in enumerate(dataloader, 0):
            global_steps=epoch*dataloader.batch_size+i
            imgs, _ = data
            imgs = imgs.to(device)
            
            # Configure input
            real_data = imgs.type(Tensor)
            label = torch.ones(imgs.shape[0], 1).to(device)
            z = torch.randn(imgs.shape[0], nz).to(device)

            if i % 5 == 0:
                # print('Discriminator')
                # ---------------------
                #  Train Discriminator
                # ---------------------

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # train with real
                optimizer_Dis.zero_grad()
                output_real = discriminator(real_data)

                # Calculate error and backpropagate
                errD_real = adversarial_loss(output_real, label)
                errD_real.backward()

                # train with fake        
                # Generate fake data
                fake_data = generator(z)
                label = torch.zeros(imgs.shape[0], 1).to(device)
                output_fake = discriminator(fake_data)
                errD_fake = adversarial_loss(output_fake, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizer_Dis.step()

            # Train the generator every n_critic steps
            if i % 1 == 0:
                # print('Generator')
                # -----------------
                #  Train Generator
                # -----------------

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                z = torch.randn(imgs.shape[0], nz).to(device)
                fake_data = generator(z)
                
                # Reset gradients
                optimizer_Gen.zero_grad()

                output = discriminator(fake_data)
                bceloss = adversarial_loss(output, torch.ones(imgs.shape[0], 1).to(device))
                errG = bceloss
                errG.backward()
                optimizer_Gen.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (epoch, opt.n_epochs, i, len(dataloader), errD.item(), errG.item(), ))
        if epoch %10==0:
            gen_dataa=gen_data(generator, discriminator, dataloader, Tensor,plot_fig_number)
            plot_buf = gen_plot(axs, gen_dataa, plot_channel_num)
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            # writer = SummaryWriter(comment='synthetic signals')
            writer.add_image('Image', image[0], global_steps)





