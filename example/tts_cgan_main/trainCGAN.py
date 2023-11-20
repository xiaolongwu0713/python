import sys
import socket

from example.tts_cgan_main.TransCGAN_model import Generator_TTSCGAN

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

from pathlib import Path
from natsort import realsorted

from gesture.DA.tts_cgan_main import cfg
from gesture.DA.tts_cgan_main.DataLoader import *
from gesture.DA.tts_cgan_main.TransCGAN_model import *
from gesture.DA.tts_cgan_main.cgan_functions import train, LinearLrDecay, load_params, copy_params
from gesture.DA.tts_cgan_main.utils import set_log_dir, save_checkpoint, create_logger, mydataset_tts_GAN

import torch
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor


def main():
    args = cfg.parse_args()
    sids = [ 17, 18, 29, 32, 41] #read_good_sids() # read_sids()
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed=False

    ngpus_per_node = torch.cuda.device_count()
    for sid in sids:
        sid=10
        args.sid=sid
        main_worker(args.gpu, ngpus_per_node, args)
        sys.exit()

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.batch_size=20
    args.latent_dim=128
    args.max_epoch=800
    #args.sid=10
    args.grow_steps = [0, 0]
    args.distributed=False
    args.norm_method = 'std'
    args.classi=0
    args.fs=1000
    args.wind=500 # 187
    args.stride=200
    args.class_num = 5  # 5
    exp = 'seeg'

    # control training process
    args.debug = False
    args.loss_metric='wgangp'  # 'wgangp'/'standard'
    args.selected_channels = True
    continuous=args.continuous
    continuous='fresh' # fresh/resume
    if args.selected_channels:
        args.chn = 10  # 1
    else:
        args.chn=208

    # log this
    experiment_description = "sid: " + str(args.sid) + ' .' +  'Loss: '+args.loss_metric+". Use 10 channels."
    print("Training GAN on sid "+str(args.sid)+".")
    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    if exp == 'mitbih':
        gen_net = Generator_TTSCGAN(seq_len=187, channels=1, num_classes=5, latent_dim=args.latent_dim, data_embed_dim=10,
                            # latent_dim=100
                            label_embed_dim=10, depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
        dis_net = Discriminator(in_channels=1, patch_size=1, data_emb_size=50, label_emb_size=10,
                                seq_length=187, depth=3, n_classes=args.class_num)
        train_set = mitbih_train()

    else:
        gen_net = Generator_TTSCGAN(seq_len=args.wind, channels=args.chn, num_classes=args.class_num,
                            latent_dim=args.latent_dim, data_embed_dim=10,  # latent_dim=100,data_embed_dim=args.chn
                            label_embed_dim=10, depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5) # num_heads=5
        if 'convlayer' in [namei for namei, _ in gen_net.named_children()] and 1==0:
            gen_net.convlayer[0].weight.data.fill_(0.99)
            gen_net.convlayer[0].bias.data.fill_(0.0001)
        # print(gen_net)
        dis_net = Discriminator(in_channels=args.chn, patch_size=1, data_emb_size=50, label_emb_size=10, # data_emb_size=args.chn,
                                seq_length=args.wind, depth=3, num_heads=5, n_classes=args.class_num)
        train_set = mydataset_tts_GAN(args=args, norm=args.norm_method, data_mode='Train', single_class=False,
                                      classi=args.classi)
    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
    gen_net = gen_net.cuda()  # torch.nn.DataParallel(gen_net).cuda()
    dis_net = dis_net.cuda()  # torch.nn.DataParallel(dis_net).cuda()

    # 1710/batch_size=1710/32=53.4375
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,shuffle=True)  # num_workers=args.num_workers,
    args.max_epoch = args.max_epoch * args.n_critic
    args.max_iter=args.max_epoch * len(train_loader)

    # def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # create new log dir
    assert args.exp_name
    args.path_helper = set_log_dir('H:/Long/data/gesture/DA/TTSCGAN/' + str(args.sid) + '/')
    logger = create_logger(args.path_helper['log_path'])
    writer = SummaryWriter(args.path_helper['log_path'])
    checkpoint_dir=Path(writer.log_dir).parent.joinpath('Model')
    print('Log dir: ' + writer.log_dir + '.')
    #if args.rank == 0:
        #args.path_helper = set_log_dir('logs', args.exp_name)
        #logger = create_logger(args.path_helper['log_path'])
        #writer = SummaryWriter(args.path_helper['log_path'])
    #continuous='resume'
    if continuous=='resume':
        #### load model from folder generated by tensorboard #####
        path = Path(writer.log_dir)
        parient_dir=path.parent.parent.absolute()
        folder_list = realsorted([str(pth) for pth in parient_dir.iterdir()]) # if pth.suffix == '.npy'])
        #pth_folder=folder_list[-2] # previous folder
        pth_folder=r'H:\Long\data\gesture\DA\TTSCGAN\10\2022_11_29_09_57_43'
        pth_list = realsorted([str(pth) for pth in Path(pth_folder).joinpath('Model').iterdir() if pth.name.startswith('checkpoint')])
        pth_file=pth_list[-1]
        #pth_list = realsorted([str(pth) for pth in Path(pth_folder).iterdir() if pth.name.startswith('checkpoint_G_epochs_')])
        #pth_file_G = pth_list[-1]

        #### load model from folder outside of tensorboard #####
        #pth_file_D = result_dir + 'checkpoint_D_epochs_*'+ '.pth'
        #pth_file_D = os.path.normpath(glob.glob(pth_file_D)[-1]) # -1 is the largest epoch number
        #pth_file_G = result_dir + 'checkpoint_G_epochs_*' + '.pth'
        #pth_file_G = os.path.normpath(glob.glob(pth_file_G)[-1])

        checkpoint = torch.load(pth_file)
        #checkpoint_G = torch.load(pth_file)
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        #### load model #####

        start_epoch=checkpoint['epoch']
        global_steps = (start_epoch + 1) * len(train_loader)
        print('Resume training from epoch '+str(start_epoch)+'.')
    elif continuous=='fresh':
        print('Fresh training.')
        start_epoch=0
        global_steps = 0


    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': global_steps,
        #'valid_global_steps': start_epoch // args.val_freq,
    }

    writer.add_text('Experiment description', experiment_description, 0)

    # train loop
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    for epoch in range(int(start_epoch), args.max_epoch+start_epoch):

        print("Epoch: "+str(epoch)+"/"+str(args.max_epoch+start_epoch))
        lr_schedulers = (gen_scheduler, dis_scheduler) # if args.lr_decay else None # None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        #train_del(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        # plot synthetic data
        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch, fig, axs, args)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image', image[0], writer_dict['train_global_steps'])
        is_best = False
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        if epoch > 150:
            save_checkpoint({
                'epoch': epoch,
                'gen_model': args.gen_model,
                'dis_model': args.dis_model,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': best_fid,
                'path_helper': args.path_helper,
                'fixed_z': fixed_z
            }, is_best, checkpoint_dir, filename="checkpoint_"+str(epoch)+".pth")
        del avg_gen_net
        
def gen_plot(gen_net, epoch,fig, axs,args):
    fig.suptitle('Synthetic data at epoch '+str(epoch)+'.', fontsize=30)

    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 
    synthetic_labels = []
    
    for i in range(5):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, args.latent_dim))).to('cuda')
        fake_label = torch.tensor([i,]).to('cuda')#fake_label = torch.randint(0, 5, (1,)).to('cuda')
        fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()
        
        synthetic_data.append(fake_sigs)
        synthetic_labels.append(fake_label)


    for i in range(2):
        for j in range(2):
            axs[i,j].plot(synthetic_data[i*2+j][0][0][0][:])
            axs[i,j].plot(synthetic_data[i*2+j][0][1][0][:])
            axs[i,j].title.set_text(synthetic_labels[i*2+j].item())
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    fig.suptitle(f'', fontsize=30)
    axs[0, 0].clear()
    axs[0, 1].clear()
    axs[1, 0].clear()
    axs[1, 1].clear()

    return buf

if __name__ == '__main__':
    main()
