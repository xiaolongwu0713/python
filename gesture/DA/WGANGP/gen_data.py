'''
use CUDA will cause OOM issue. Use CPU instead.
'''

from common_dl import LongTensor
from gesture.DA.GAN.gan import SEEG_CNN_Generator_10channels
from gesture.DA.cTGAN.models import Generator
from gesture.DA.cTGAN.main import parse_args
from copy import deepcopy
import torch
import numpy as np

# CWGANGP
sid=10
time_stamp2='2024_04_11_15_17_12'
ckpt_epoch=700
trial_num=304
latent_dims=512
generator = SEEG_CNN_Generator_10channels(10,latent_dims)
result_dir='D:/tmp/python/gesture/DA/CWGANGP/sid' + str(sid) + '/'+time_stamp2+'/'
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
    filename = result_dir + 'Samples/class_' + str(i) + '.npy'
    np.save(filename, tmp2)

