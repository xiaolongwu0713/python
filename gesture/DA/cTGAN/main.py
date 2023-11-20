from gesture.DA.cTGAN.models import *
from gesture.DA.cTGAN.ctgan import train, LinearLrDecay, load_params, copy_params, cur_stages
from gesture.DA.cTGAN.utils import set_log_dir, save_checkpoint, create_logger, mydataset

from pathlib import Path
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--seed', default=12345, type=int,help='seed for initializing training. ')
    parser.add_argument('--max_epoch',type=int,default=500,help='number of epochs of training')
    parser.add_argument('--max_iter',type=int,default=None,help='set the max iteration number')
    parser.add_argument('-gen_bs','--gen_batch_size',type=int,default=32,help='size of the batches')
    parser.add_argument('-dis_bs','--dis_batch_size',type=int,default=64,help='size of the batches')
    parser.add_argument('-bs','--batch_size',type=int,default=64,help='size of the batches to load dataset')
    parser.add_argument('--g_lr',type=float,default=0.0002,help='adam: gen learning rate')
    parser.add_argument('--d_lr',type=float,default=0.0002,help='adam: disc learning rate')
    parser.add_argument('--lr_decay',action='store_true',help='learning rate decay or not')
    parser.add_argument('--beta1',type=float,default=0.0,help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float,default=0.9,help='adam: decay of first order momentum of gradient')
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
    parser.add_argument('--ema', type=float, default=0.995,help='ema')
    parser.add_argument('--ema_warmup', type=float, default=0.,help='ema warm up')
    parser.add_argument('--ema_kimg', type=int, default=500,help='ema thousand images')

    opt = parser.parse_args()

    return opt

def main():
    args = parse_args()
    
    random_seed=12345
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Simply call main_worker function
    main_worker(args)
        
def main_worker(args):
    args.grow_steps = [0, 0]

    # import network
    gen_net = Generator(seq_len=500, channels=10, num_classes=5, latent_dim=args.latent_dim, data_embed_dim=10,
                        label_embed_dim=10 ,depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
    dis_net = Discriminator(in_channels=10, patch_size=1, data_emb_size=50, label_emb_size=10, seq_length = 500, depth=3, n_classes=5)

    gen_net = gen_net.cuda()
    dis_net = dis_net.cuda()

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                    args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                    args.d_lr, (args.beta1, args.beta2))

    args.max_iter=args.max_epoch * 54
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter)

    args.norm_method='std'
    args.classi=0
    args.sid=10
    args.fs=1000
    args.wind=500
    args.chn=10
    args.stride=200
    args.selected_channels = True
    args.batch_size=32
    train_set = mydataset(args=args, norm=args.norm_method, data_mode='Train', single_class=False,
                                  classi=args.classi)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net) # model parameter to list
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    args.path_helper = set_log_dir('H:/Long/data/gesture/DA/TTSCGAN/' + str(args.sid) + '/')
    writer = SummaryWriter(args.path_helper['log_path'])
    checkpoint_dir = Path(writer.log_dir).parent.joinpath('Model')
    print('Log dir: ' + writer.log_dir + '.')

    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
    }

    # training
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        print("Epoch: " + str(epoch) + "/" + str(args.max_epoch + start_epoch))
        args.lr_decay=True
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank==0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank==0 else 0

        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)

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
        }, checkpoint_dir, filename="checkpoint_"+str(epoch)+".pth")
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
