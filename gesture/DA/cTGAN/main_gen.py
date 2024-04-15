'''
use CUDA will cause OOM issue. Use CPU instead.
'''
import argparse
import sys, os
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])

from common_dl import LongTensor
from gesture.DA.GAN.gan import SEEG_CNN_Generator_10channels
from gesture.DA.cTGAN.models import Generator
from gesture.DA.cTGAN.main_train import parse_args
from copy import deepcopy
import torch
import numpy as np
from pre_all import running_from_IDE,running_from_CMD

if running_from_CMD:
    sid = int(float(sys.argv[1]))
    cv=int(float(sys.argv[2]))
    #time_stamp=sys.argv[3]
    #ckpt_epoch=int(sys.argv[4])
else:
    sid=10
    cv=1
from gesture.config import time_stamps,ckpt_epochs
time_stamp=time_stamps[0][cv-1]
ckpt_epoch=ckpt_epochs[0][cv-1]

## CTGAN
latent_dim=128
gen_net = Generator(seq_len=500, channels=10, num_classes=5, latent_dim=latent_dim, data_embed_dim=10,
                        label_embed_dim=10 ,depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
gen_net = gen_net #.cuda()
avg_gen_net = deepcopy(gen_net).cpu()

result_dir='D:/tmp/python/gesture/DA/cTGAN/sid' + str(sid) + '/cv'+str(cv)+'/'+time_stamp+'/'
print('Result dir:'+result_dir+'.')
checkpoint_file=result_dir +'Model/checkpoint_'+str(ckpt_epoch)+'.pth'
cpu=torch.device('cpu')
checkpoint = torch.load(checkpoint_file,map_location=cpu)
gen_net.load_state_dict(checkpoint['gen_state_dict'])
gen_net.eval()
#avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])

for i in range(5):
    train_set_size=304 # 304*5=1520
    noise = torch.FloatTensor(np.random.normal(0, 1, (train_set_size, latent_dim)))
    label = torch.tensor([i, ]*train_set_size)
    tmp = gen_net(noise, label).detach().numpy().squeeze() # torch.Size([32, 10, 1, 500])
    #gen_data.append(tmp)
    #gen_data=np.concatenate(gen_data,axis=0)
    #gen_data_avg = avg_gen_net(noise, label).to('cpu').detach().numpy()

    filename=result_dir+'Samples/class'+str(i)+'_cv'+str(cv)+'.npy'
    np.save(filename,tmp)

    #filename = result_dir + 'Samples/class_' + str(i) + '_avg.npy'
    #np.save(filename, gen_data_avg)
