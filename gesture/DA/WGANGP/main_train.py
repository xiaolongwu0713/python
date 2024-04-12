import argparse
import sys, os
import socket

from gesture.DA.cTGAN.ctgan import LinearLrDecay

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])

import PIL
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm
from gesture.DA.metrics.visualization import visualization
from gesture.DA.cTGAN.utils import save_checkpoint
import dateutil, pytz
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gesture.channel_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel
from gesture.utils import *
from gesture.config import *
from gesture.DA import cfg_cmd
from gesture.DA.GAN.gan import gan, SEEG_CNN_Generator_10channels, weights_init, compute_gradient_penalty, \
    gen_data_wgangp_, gen_plot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', default=10, type=int)
    parser.add_argument('--fs', default=1000, type=int)
    parser.add_argument('--wind', default=500, type=int)
    parser.add_argument('--stride', default=100, type=int)
    parser.add_argument('--epochs',type=int,default=500,help='number of epochs of training')
    parser.add_argument('--load_path',type=str,help='The reload model path')
    parser.add_argument('--gen_method', type=str, default='CWGANGP',help='The reload model path')
    parser.add_argument('--selected_channels', type=str, default='Yes', help='use selected channels or not')
    opt = parser.parse_args()
    return opt

args = parse_args()
sid, fs, wind, stride, gen_method, epochs, selected_channels = args.sid, args.fs, args.wind, args.stride, args.gen_method, args.epochs, args.selected_channels

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True


now = datetime.now(pytz.timezone('Asia/Shanghai'))  # dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

#args = cfg_cmd.parse_args()
latent_dims = 512
#args.load_path='D:/tmp/python/gesture/DA/CWGANGP/10/2024_04_11_11_34_05/Model/checkpoint_290.pth'
# print(str(sid)+','+str(fs)+','+str(wind)+','+str(stride)+','+gen_method+','+continuous+','+str(epochs)+','+str(selected_channels))

class_number = 5
if selected_channels:
    selected_channels, acc = get_selected_channel_gumbel(args.sid, 10)  # 10 selected channels
else:
    selected_channels = None
#
# pre_fix = tmp_dir + 'DA/' + gen_method + '/' + str(sid) + '/' + timestamp + '/'
# save_gen_dir = pre_fix + 'Sample/'
# # result_dir = result_dir + 'DA/' + gen_method+ '/'+ str(sid) + '/'
# model_dir = pre_fix + 'Model/'
# log_dir = pre_fix + 'Log/'
# writer = SummaryWriter(log_dir)
#
# if not os.path.exists(pre_fix):
#     os.makedirs(pre_fix)
# if not os.path.exists(save_gen_dir):
#     os.makedirs(save_gen_dir)
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# print('Result dir: ' + pre_fix + '.')
# print('Generate data in dir: ' + save_gen_dir + '.')

# can try to use selected channels
if gen_method == 'DCGAN' or gen_method == 'CWGANGP' or gen_method == 'CWGAN':
    norm_method = 'std'  # minmax
else:
    norm_method = 'std'  # minmax/std

# def train(sid,continuous,train_loader,chn_num):

batch_size = 32
test_epochs, val_epochs, train_epochs, scaler = read_data_split_function(sid, fs, selected_channels=selected_channels,
                                                                         scaler=norm_method)
X_train, y_train, X_val, y_val, X_test, y_test = windowed_data(train_epochs, val_epochs, test_epochs, wind, stride)
total_trials = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val))
train_set = myDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
one_batch = next(iter(train_loader))[0]  # torch.Size([32, 208, 500])
chn_num = one_batch.shape[1]
labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (304, 10, 500)

print("Generate data using " + gen_method + ".")
#gan(gen_method, writer, sid, continuous, chn_num, class_number, wind, train_loader, epochs, latent_dims)

learning_rate = 0.0002  # 0.0001/2
weight_decay = 0.001
dropout_level = 0.05
class_number = 5
wind = 500
lambda_gp=10
adversarial_loss = nn.BCEWithLogitsLoss()
#batch_size=train_loader.batch_size  # last batch is not the same
generator = SEEG_CNN_Generator_10channels(chn_num,latent_dims).to(device)  # SEEG_CNN_Generator().to(device)
from gesture.DA.GAN.models import deepnet
discriminator = deepnet(gen_method,chn_num, 1, wind).to(device)  # SEEG_CNN_Discriminator().to(device)
#discriminator = Discriminator(chn_num).to(device)
# Optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
#gen_scheduler = LinearLrDecay(optimizer_G, learning_rate, 0.0001, 0, args.max_iter)
#dis_scheduler = LinearLrDecay(optimizer_D, learning_rate, 0.0001, 0, args.max_iter)

if args.load_path:
    print(f'=> resuming from {args.load_path}')
    assert os.path.exists(args.load_path)
    checkpoint_file = os.path.join(args.load_path)
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    train_global_steps = checkpoint['train_global_steps']

    discriminator.load_state_dict(checkpoint['dis_state_dict'])
    optimizer_G.load_state_dict(checkpoint['gen_optimizer'])
    optimizer_D.load_state_dict(checkpoint['dis_optimizer'])

    generator.load_state_dict(checkpoint['gen_state_dict'])

    pre_epochs = checkpoint['epoch']
    global_steps = (pre_epochs + 1) * len(train_loader)
    print('Resume training. The last training epoch is ' + str(pre_epochs) + '.')

    args.path_helper = checkpoint['path_helper']
    # logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {pre_epochs})')
    writer = SummaryWriter(args.path_helper['log_path'])
    log_path=args.path_helper['log_path']
    del checkpoint
else:
    print(f'=> Fresh training. ')
    args.path_helper = {}  # = set_log_dir('D:/data/BaiduSyncdisk/gesture/DA/cTGAN/' + str(args.sid) + '/')
    # path_dict = {}
    prefix = tmp_dir + 'DA/CWGANGP/sid' + str(args.sid) + '/' + timestamp + '/'
    os.makedirs(prefix)
    args.path_helper['prefix'] = prefix

    ckpt_path = prefix + 'Model/'  # /os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    args.path_helper['ckpt_path'] = ckpt_path

    log_path = prefix + 'Log/'  # os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    args.path_helper['log_path'] = log_path

    sample_path = prefix + 'Samples'  # os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    args.path_helper['sample_path'] = sample_path

    writer = SummaryWriter(args.path_helper['log_path'])
    pre_epochs = 0
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    global_steps = 0

print('Log path: '+log_path+'.')

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
    for i, data in enumerate(tqdm(train_loader)):
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
            if gen_method == 'CWGANGP':
                fake_label = real_label  # , for GP calculation
            elif gen_method == 'CDCGAN':
                fake_label = torch.randint(0, 5, real_label.shape).to(device)
            fake_data = generator(z, fake_label)  # torch.Size([32, 208, 500])

            # TODO: test using 0.9 and 0.1 for valid and fake label
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            real_validity = discriminator(real_data, real_label)  # scalar value torch.Size([32, 1])
            fake_validity = discriminator(fake_data, fake_label)  # torch.Size([32, 1])
            ####  forward pass ####

            if gen_method == 'CWGANGP':
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data,fake_label)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                d_loss_item = d_loss.item()
                d_loss.backward(retain_graph=False)#d_loss.backward(retain_graph=True) # multiple backward propagation(error prone)
                optimizer_D.step()
                writer.add_scalar('d_loss', d_loss_item, global_steps)
            elif gen_method=='CDCGAN':
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
            if gen_method == 'CWGANGP':
                fake_label = real_label  # , for GP calculation
            elif gen_method == 'CDCGAN':
                fake_label = torch.randint(0, 5, real_label.shape).to(device)
            fake_data = generator(z, fake_label)  # torch.Size([32, 208, 500])

            # TODO: test using 0.9 and 0.1 for valid and fake label
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            real_validity = discriminator(real_data, real_label)  # scalar value torch.Size([32, 1])
            fake_validity = discriminator(fake_data, fake_label)  # torch.Size([32, 1])
            ####  forward pass ####

            if gen_method=='CWGANGP':
                g_loss = -torch.mean(fake_validity)
                g_loss_item = g_loss.item()
                g_loss.backward()
                optimizer_G.step()
                writer.add_scalar('g_loss', g_loss_item, global_steps)
            elif gen_method == 'CDCGAN':
                g_loss = adversarial_loss(real_validity, valid)
                g_loss_item = g_loss.item()
                g_loss.backward()
                optimizer_G.step()
                writer.add_scalar('g_loss', g_loss_item, global_steps)

        d_losses.append(d_loss_item)
        g_losses.append(g_loss_item)
        global_steps += 1
    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
        epoch, pre_epochs + epochs, i, len(train_loader), d_loss_item, g_loss_item))

    if (epoch + 1) % 1 == 0:
        # def gen_data_wgangp(sid, chn_num, class_num, wind_size, result_dir, num_data_to_generate=500):
        gen_data = gen_data_wgangp_(generator, latent_dims,num_data_to_generate=304)  # (batch_size, 208, 500)
        #plot_buf = gen_plot(axs, gen_data, plot_channel_num)
        for i in range(2):  # 4 plots
            for j in range(2):
                axs[i, j].clear()
                for k in range(plot_channel_num):
                    axs[i, j].plot(gen_data[i * 2 + j, k, :])
        plt.figure(fig)
        plt.savefig('del_figure.png')
        image = PIL.Image.open('../del_figure.png')
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image/raw', image[0], global_steps)

        ff = visualization([gen_data.transpose(0, 2, 1), X_train_class0.transpose(0, 2, 1)], 'tSNE', labels,
                                  display=False,epoch=epoch)
        plt.figure(ff)
        plt.savefig('del_figure2.png')
        image2 = PIL.Image.open('../del_figure2.png')
        image2 = ToTensor()(image2).unsqueeze(0)
        writer.add_image('Image/tSNE', image2[0], global_steps)

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
        if 1==1: #epoch>100:
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': generator.state_dict(),
                'dis_state_dict': discriminator.state_dict(),
                'gen_optimizer': optimizer_G.state_dict(),
                'dis_optimizer': optimizer_D.state_dict(),
                'path_helper': args.path_helper,
                'train_global_steps': global_steps,
            }, args.path_helper['ckpt_path'], filename="checkpoint_" + str(epoch) + ".pth")

            #savepath_D = writer.log_dir + 'checkpoint_D_epochs_' + str(epoch + pre_epochs) + '.pth'
            #savepath_G = writer.log_dir + 'checkpoint_G_epochs_' + str(epoch + pre_epochs) + '.pth'
            #torch.save(state_D, savepath_D)
            #torch.save(state_G, savepath_G)
