import os
import sys
import socket

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])

from gesture.DA.cTGAN.models import *
from gesture.DA.cTGAN.ctgan import train, LinearLrDecay, load_params, copy_params, cur_stages, gradient_penalty
from gesture.DA.cTGAN.utils import save_checkpoint, create_logger, mydataset
from tqdm import tqdm
import torch
import torch.utils.data.distributed
from torch.utils import data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import argparse
from gesture.config import tmp_dir
from datetime import datetime
import pytz
now = datetime.now(pytz.timezone('Asia/Shanghai'))
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', default=10, type=int)
    parser.add_argument('--cv', type=int)
    parser.add_argument('--world-size', default=-1, type=int)
    #parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--seed', default=12345, type=int,help='seed for initializing training. ')
    parser.add_argument('--max_epoch',type=int,default=500,help='number of epochs of training')
    parser.add_argument('--max_iter',type=int,default=None,help='set the max iteration number')
    parser.add_argument('-gen_bs','--gen_batch_size',type=int,default=32,help='size of the batches')
    parser.add_argument('-dis_bs','--dis_batch_size',type=int,default=64,help='size of the batches')
    parser.add_argument('-bs','--batch_size',type=int,default=64,help='size of the batches to load dataset')
    parser.add_argument('--g_lr',type=float,default=0.0002,help='adam: gen learning rate')
    parser.add_argument('--d_lr',type=float,default=0.0002,help='adam: disc learning rate')
    parser.add_argument('--lr_decay',action='store_true',help='learning rate decay or not')
    #parser.add_argument('--beta1',type=float,default=0.0,help='adam: decay of first order momentum of gradient')
    #parser.add_argument('--beta2',type=float,default=0.9,help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim',type=int,default=128,help='dimensionality of the latent space')
    parser.add_argument('--img_size',type=int,default=32,help='size of each image dimension')
    parser.add_argument('--channels',type=int,default=3,help='number of image channels')
    parser.add_argument('--n_critic',type=int,default=1,help='number of training steps for discriminator per iter')
    parser.add_argument('--print_freq',type=int,default=100,help='interval between each verbose')
    parser.add_argument('--load_path',type=str,help='The reload model path')
    parser.add_argument('--class_name',type=str,help='The class name to load in UniMiB dataset')
    parser.add_argument('--max_search_iter', type=int, default=90,help='max search iterations of this algorithm')
    parser.add_argument('--hid_size', type=int, default=100,help='the size of hidden vector')
    parser.add_argument('--baseline_decay', type=float, default=0.9,help='baseline decay rate in RL')
    parser.add_argument('--loss', type=str, default="hinge",help='loss function')
    parser.add_argument('--n_classes', type=int, default=0,help='classes')
    parser.add_argument('--phi', type=float, default=1,help='wgan-gp phi')
    parser.add_argument('--grow_steps', nargs='+', type=int,help='the vector of a discovered architecture')
    parser.add_argument('--patch_size', type=int, default=4,help='Discriminator Depth')
    parser.add_argument('--fid_stat', type=str, default="None",help='Discriminator Depth')
    parser.add_argument('--d_heads', type=int, default=4,help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.,help='dropout ratio')
    #parser.add_argument('--ema', type=float, default=0.995,help='ema')
    #parser.add_argument('--ema_warmup', type=float, default=0.,help='ema warm up')
    #parser.add_argument('--ema_kimg', type=int, default=500,help='ema thousand images')
    opt = parser.parse_args()
    return opt

def main():
    args = parse_args()
    args.load_path='D:/tmp/python/gesture/DA/cTGAN/sid10/2024_04_10_16_09_06/Model/checkpoint_290.pth'
    args.norm_method = 'std'
    #sid=10
    #args.sid = sid
    args.fs = 1000
    args.wind = 500
    args.chn = 10
    args.stride = 200
    args.selected_channels = True
    args.batch_size = 32
    random_seed=12345
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Sid:'+str(args.sid)+'.')
    args.grow_steps = [0, 0]

    # define network
    gen_net = Generator(seq_len=500, channels=10, num_classes=5, latent_dim=args.latent_dim, data_embed_dim=10,
                        label_embed_dim=10 ,depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
    dis_net = Discriminator(in_channels=10, patch_size=1, data_emb_size=50, label_emb_size=10, seq_length = 500, depth=3, n_classes=5)

    gen_net = gen_net.cuda()
    dis_net = dis_net.cuda()

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),args.g_lr)
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),args.d_lr)

    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        train_global_steps=checkpoint['train_global_steps']
        best_fid = checkpoint['best_fid']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        #fixed_z = checkpoint['fixed_z']

        args.path_helper = checkpoint['path_helper']
        if 1==1:
            args.path_helper = {}
            prefix= 'D:/tmp/python/gesture/DA/cTGAN/sid10/2024_04_10_16_09_06/'
            args.path_helper['prefix']=prefix
            ckpt_path = prefix + 'Model/'  # /os.path.join(prefix, 'Model')
            args.path_helper['ckpt_path'] = ckpt_path
            log_path = prefix + 'Log/'  # os.path.join(prefix, 'Log')
            args.path_helper['log_path'] = log_path
            sample_path = prefix + 'Samples'  # os.path.join(prefix, 'Samples')
            args.path_helper['sample_path'] = sample_path
        #logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path'])
        del checkpoint
    else:
        start_epoch = 0
        train_global_steps=0
        best_fid = 1e4
        args.path_helper = {}  # = set_log_dir('D:/data/BaiduSyncdisk/gesture/DA/cTGAN/' + str(args.sid) + '/')
        # path_dict = {}
        os.makedirs(tmp_dir, exist_ok=True)

        # set log path
        # exp_path = os.path.join(root_dir, exp_name)
        prefix = tmp_dir + 'DA/cTGAN/sid'+str(args.sid)+'/' + timestamp + '/'
        os.makedirs(prefix)
        args.path_helper['prefix'] = prefix

        # set checkpoint path
        ckpt_path = prefix + 'Model/'  # /os.path.join(prefix, 'Model')
        os.makedirs(ckpt_path)
        args.path_helper['ckpt_path'] = ckpt_path

        log_path = prefix + 'Log/'  # os.path.join(prefix, 'Log')
        os.makedirs(log_path)
        args.path_helper['log_path'] = log_path

        # set sample image path for fid calculation
        sample_path = prefix + 'Samples'  # os.path.join(prefix, 'Samples')
        os.makedirs(sample_path)
        args.path_helper['sample_path'] = sample_path

        writer = SummaryWriter(args.path_helper['log_path'])
        checkpoint_dir = args.path_helper['ckpt_path']  # Path(writer.log_dir).parent.joinpath('Model')
        print('Log dir: ' + writer.log_dir + '.')

    writer_dict = {
        'writer': writer,
        'train_global_steps': train_global_steps,
    }

    args.max_iter=args.max_epoch * 54
    end_lr=0.00012
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, end_lr, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, end_lr, 0, args.max_iter)


    train_set = mydataset(args=args, norm=args.norm_method, data_mode='Train',cv_idx=args.cv) # sid10:1710
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net) # model parameter to list
    del avg_gen_net


    # training
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        print("Epoch: " + str(epoch) + "/" + str(args.max_epoch + start_epoch))
        args.lr_decay=True
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank==0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank==0 else 0

        #train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)
        #writer = writer_dict['writer']
        gen_step = 0
        cls_criterion = nn.CrossEntropyLoss()
        lambda_cls = 1
        lambda_gp = 10

        gen_net.train()
        dis_net.train()

        for iter_idx, (real_imgs, real_img_labels) in enumerate(tqdm(train_loader)):
            global_steps = writer_dict['train_global_steps']

            # Adversarial ground truths
            real_imgs = real_imgs.type(torch.cuda.FloatTensor).cuda()
            #         real_img_labels = real_img_labels.type(torch.IntTensor)
            real_img_labels = real_img_labels.type(torch.LongTensor)
            real_img_labels = real_img_labels.cuda()

            # Sample noise as generator input
            noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).cuda()
            fake_img_labels = torch.randint(0, 5, (real_imgs.shape[0],)).cuda()

            #  Train Discriminator

            dis_net.zero_grad()
            r_out_adv, r_out_cls = dis_net(real_imgs)
            fake_imgs = gen_net(noise, fake_img_labels)
            f_out_adv, f_out_cls = dis_net(fake_imgs)

            # gradient penalty
            alpha = torch.rand(real_imgs.size(0), 1, 1, 1).cuda()  # bh, C, H, W
            x_hat = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
            out_src, _ = dis_net(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, args)

            d_real_loss = -torch.mean(r_out_adv)
            d_fake_loss = torch.mean(f_out_adv)
            d_adv_loss = d_real_loss + d_fake_loss

            d_cls_loss = cls_criterion(r_out_cls, real_img_labels)

            d_loss = d_adv_loss + lambda_cls * d_cls_loss + lambda_gp * d_loss_gp
            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()

            writer.add_scalar('d_loss', d_loss.item(), writer_dict['train_global_steps'])

            #  Train Generator
            gen_net.zero_grad()

            gen_imgs = gen_net(noise, fake_img_labels)
            g_out_adv, g_out_cls = dis_net(gen_imgs)

            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = cls_criterion(g_out_cls, fake_img_labels)
            g_loss = g_adv_loss + lambda_cls * g_cls_loss
            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()

            # adjust learning rate
            if lr_schedulers:
                gen_scheduler, dis_scheduler = lr_schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # # moving average weight
            # ema_nimg = args.ema_kimg * 1000
            # cur_nimg = args.dis_batch_size * args.world_size * global_steps
            # if args.ema_warmup != 0:
            #     ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            #     ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
            # else:
            #     ema_beta = args.ema
            #
            # # moving average weight
            # for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            #     cpu_p = deepcopy(p)
            #     avg_p = avg_p * ema_beta + ((1 - ema_beta) * cpu_p.cpu().data)
            #     del cpu_p

            writer.add_scalar('g_loss', g_loss.item(), writer_dict['train_global_steps'])
            gen_step += 1

            if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
                tqdm.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                    (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(),
                     g_loss.item(),
                     ema_beta))
            writer_dict['train_global_steps'] = global_steps + 1


        # plot the generated data
        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch, fig, axs, args)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image', image[0], writer_dict['train_global_steps'])
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper,
            'train_global_steps':global_steps,
        }, args.path_helper['ckpt_path'], filename="checkpoint_"+str(epoch)+".pth")
        del avg_gen_net

def gen_plot(gen_net, epoch, fig, axs, args):
    fig.suptitle('Generated data at epoch ' + str(epoch) + '.', fontsize=30)

    synthetic_data = []
    synthetic_labels = []

    for i in range(5):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, args.latent_dim))).to('cuda')
        fake_label = torch.tensor([i, ]).to('cuda')
        fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()

        synthetic_data.append(fake_sigs)
        synthetic_labels.append(fake_label)

    for i in range(2):
        for j in range(2):
            axs[i, j].plot(synthetic_data[i * 2 + j][0][0][0][:])
            axs[i, j].plot(synthetic_data[i * 2 + j][0][1][0][:])
            axs[i, j].title.set_text(synthetic_labels[i * 2 + j].item())
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
