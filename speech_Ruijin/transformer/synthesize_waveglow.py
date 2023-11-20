'''
below is a must if you want to use waveglow
mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256,
                               win_length=1024,
                               n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                               mel_fmax=8000)

'''

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
    sys.path.extend(['C:/Users/xiaowu/mydrive/python/'])

from pre_all import running_from_CMD
from speech_Ruijin.config import data_dir
import math
import numpy as np
from common_dl import device, set_random_seeds
from speech_Ruijin.baseline_linear_regression.extract_features import dataset
from speech_Ruijin.transformer.opt import opt_SingleWordProductionDutch as opt
import torch
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

set_random_seeds(99999)
if running_from_CMD:
    sid = int(float(sys.argv[1]))
    dataname = sys.argv[2] #'SingleWordProductionDutch'
    time_stamp=sys.argv[3]
    folder_date=time_stamp
else:
    sid=3
    dataname = 'SingleWordProductionDutch'
fig,ax=plt.subplots()
audio_sr=22050

############
sf_EEG = opt['sf_EEG']
mel_bins = opt['mel_bins']
stride = opt['stride']
step_size = opt['step_size']
model_order = opt['model_order']
use_the_official_tactron_with_waveglow=opt['use_the_official_tactron_with_waveglow']
if use_the_official_tactron_with_waveglow:
    target_SR = 22050
    frameshift = 256 / target_SR  # 256 steps in 0.011609977324263039 s
    winL = 1024 / target_SR
else:
    winL = opt['winL']
    frameshift = opt['frameshift']
#winL = opt['winL']
#frameshift = opt['frameshift']
win = math.ceil(opt['win'] / frameshift)  # int: steps
history = math.ceil(opt['history'] / frameshift)  # int(opt['history']*sf_EEG) # int: steps
baseline_method = opt['baseline_method']
stride_test = opt['stride_test']
#########
result_dir = data_dir + 'seq2seq_transformer/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(
                sid) + '/' + folder_date + '/'

waveglow=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()
# or compute the mel using tactron
from speech_Ruijin.utils import TacotronSTFT
mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256,
                               win_length=1024,
                               n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                               mel_fmax=8000)


x, y, audio_original = dataset(dataset_name=dataname, sid=sid, melbins=mel_bins, stacking=False, winL=winL, target_SR =target_SR,frameshift=frameshift,
                               return_original_audio=True,use_the_official_tactron_with_waveglow=use_the_official_tactron_with_waveglow)
import resampy
audio_original = resampy.resample(audio_original, 48000, 22050)
lenx = x.shape[0]
leny = y.shape[0]
len_audio=audio_original.shape[0]
train_x = x[:int(lenx * 0.8), :]
val_x = x[int(lenx * 0.8):int(lenx * 0.9), :]
test_x = x[int(lenx * 0.9):, :]
train_y = y[:int(leny * 0.8), :]
val_y = y[int(leny * 0.8):int(leny * 0.9), :]
test_y = y[int(leny * 0.9):, :] #(3003, 80)
audio=audio_original[int(len_audio * 0.9):]

filename=result_dir+'audio_original.wav'
wavfile.write(filename, audio_sr, audio)

#audio=np.clip(audio/np.max(np.abs(audio)),-1,1)
mel_original = mel_transformer.mel_spectrogram(torch.from_numpy(audio)[np.newaxis, :].to(torch.float32)).squeeze()
with torch.no_grad():
    audio_original_recon = waveglow.infer(mel_original[None,:,:].to(device)) # mel: torch.Size([1, 80, 185])
audio_original_recon=audio_original_recon.to('cpu').squeeze().numpy()
filename=result_dir+'original_recon.wav'
wavfile.write(filename, audio_sr, audio_original_recon)

#filename='H:/Long/data/speech_Ruijin/seq2seq_transformer/SingleWordProductionDutch/mel_80/sid_3/2023_09_03_12_05/mel_pred.npy'
mel_pred=np.load(result_dir+'mel_pred.npy') # (2555, 80)
mel_pred=torch.from_numpy(mel_pred.transpose()).float()
with torch.no_grad():
    audio_pred_recon = waveglow.infer(mel_pred[None,:,:].to(device)) # mel: torch.Size([1, 80, 185])
audio_pred_recon=audio_pred_recon.to('cpu').squeeze().numpy()
audio_pred_recon_scaled=np.clip(audio_pred_recon/np.max(np.abs(audio_pred_recon)),-0.6,0.5)
filename=result_dir+'audio_pred_recon.wav'
wavfile.write(filename,audio_sr,audio_pred_recon_scaled)

fig,ax=plt.subplots(3,1)
ax[0].plot(audio)
ax[1].plot(audio_original_recon)
ax[2].plot(audio_pred_recon_scaled)
filename=result_dir+'audio_recon.pdf'
fig.savefig(filename)
