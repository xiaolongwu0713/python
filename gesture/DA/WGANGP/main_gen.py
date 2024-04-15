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

import os,sys
from common_dl import LongTensor
from gesture.DA.GAN.gan import SEEG_CNN_Generator_10channels
from gesture.DA.cTGAN.models import Generator
from gesture.DA.cTGAN.main_train import parse_args
from copy import deepcopy
import torch
import numpy as np

if os.environ.get('PYCHARM_HOSTED'):
    running_from_IDE=True
    running_from_CMD = False
    print("Running from IDE.")
else:
    running_from_CMD = True
    running_from_IDE = False
    print("Running from CMD.")

if running_from_CMD:
    sid = int(float(sys.argv[1]))
    cv=int(float(sys.argv[2]))
    #time_stamp=sys.argv[3]
    #ckpt_epoch=int(sys.argv[4])
else:
    sid=10
    cv=1
    #time_stamp='2024_04_12_14_49_29'
    #ckpt_epoch=444

from gesture.config import time_stamps,ckpt_epochs
# CWGANGP
time_stamp=time_stamps[1][cv-1]#['2024_04_12_14_49_29',]
ckpt_epoch=ckpt_epochs[1][cv-1] #[333,]

trial_num=304
latent_dims=512
generator = SEEG_CNN_Generator_10channels(10,latent_dims)
result_dir='D:/tmp/python/gesture/DA/CWGANGP/sid' + str(sid) + '/cv'+str(cv)+'/'+time_stamp+'/'
checkpoint_file=result_dir +'Model/checkpoint_'+str(ckpt_epoch)+'.pth'
cpu=torch.device('cpu')
checkpoint = torch.load(checkpoint_file,map_location=cpu)
generator.load_state_dict(checkpoint['gen_state_dict'])
generator.eval()

# generate for class 0
for i in range(5): # 5
    z = torch.randn(trial_num, latent_dims)
    gen_label = torch.ones((trial_num, 1)).type(LongTensor).to('cpu') * i
    tmp2 = generator(z, gen_label)  #
    tmp2 = tmp2.detach().numpy()  # ([304, 10, 500])
    filename = result_dir + 'Samples/class' + str(i) + '_cv'+str(cv)+'.npy'
    np.save(filename, tmp2)
print("Result dir: "+result_dir+'Samples/.')

