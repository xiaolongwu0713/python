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

sid=10
time_stamp='2024_04_10_16_09_06'
ckpt_epoch=390
latent_dim=128
gen_net = Generator(seq_len=500, channels=10, num_classes=5, latent_dim=latent_dim, data_embed_dim=10,
                        label_embed_dim=10 ,depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
gen_net = gen_net #.cuda()
avg_gen_net = deepcopy(gen_net).cpu()

result_dir='D:/tmp/python/gesture/DA/cTGAN/sid' + str(sid) + '/'+time_stamp+'/'
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

    filename=result_dir+'Samples/class_'+str(i)+'.npy'
    np.save(filename,tmp)

    #filename = result_dir + 'Samples/class_' + str(i) + '_avg.npy'
    #np.save(filename, gen_data_avg)


# CWGANGP
sid=10
time_stamp2='2024_04_11_15_17_12'
ckpt_epoch=308
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

