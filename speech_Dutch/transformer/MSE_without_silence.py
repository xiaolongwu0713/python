'''
This script was used to calculate MSE and CC loss using data from speech production (without silent interval).
Manual extraction of the task interval and calculate the MES/CC: CC declined from 0.9 to 0.8 while MSE increased from 0.07 to 0.13;
'''
from pre_all import computer
from speech_Dutch.seq2seq.opt import opt_SingleWordProductionDutch as opt
from speech_Dutch.config import data_dir
import numpy as np
import math
from scipy.stats import pearsonr
from torch import nn
import torch
import matplotlib.pyplot as plt

sid = 3
dataname='SingleWordProductionDutch'
time_stamp='dummy'
mel_bins = 80


###############
use_the_official_tactron_with_waveglow = opt['use_the_official_tactron_with_waveglow']
window_eeg=opt['window_eeg']
winL = opt['winL']
target_SR=opt['target_SR']
frameshift = opt['frameshift']
#mel_bins = opt['mel_bins']
win = math.ceil(opt['win'] / frameshift)  # int: steps
history = math.ceil(opt['history'] / frameshift)  # int(opt['history']*sf_EEG) # int: steps
stride = opt['stride']
stride_test = opt['stride_test']
baseline_method = opt['baseline_method']
norm_EEG = opt['norm_EEG']  # True
norm_mel = opt['norm_mel']  # False
sf_EEG = opt['sf_EEG']

embed_size = opt['embed_size']  # = 256
num_hiddens = opt['num_hiddens']  # = 256
num_layers = opt['num_layers']  # = 2
dropout = opt['dropout']  # = 0.5
lr = opt['lr']
batch_size = opt['batch_size']


if computer == 'workstation' or 'Yoga':
    result_dir = data_dir + 'seq2seq_transformer/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(sid) + '/'
elif computer == 'mac':
    result_dir = '/Users/xiaowu/tmp/models/seq2seq/'

filename = result_dir+'2023_11_20_16_15/' + 'mel_pred.npy'
pred=np.load(filename)
filename = result_dir+'2023_11_20_16_15/' + 'mel_tgt.npy'
tgt=np.load(filename)

corr = []  # transformer: mean=0.8677304214419497 seq2seq:0.8434792922232249
for specBin in range(pred.shape[1]):
    r, p = pearsonr(pred[:, specBin], tgt[:, specBin])
    corr.append(r)
mean_corr=sum(corr) / len(corr)
print('corr:'+str(mean_corr)) # 0.9115962016129762

criterion = nn.MSELoss()
loss = criterion(torch.from_numpy(pred),torch.from_numpy(tgt))  # 40:10.6792; 80: tensor(18.4728, dtype=torch.float64)
from sklearn.metrics import mean_squared_error
loss=mean_squared_error(pred, tgt)
print('MSE:'+str(loss)) # 0.07475251031818642

fig,ax=plt.subplots(2,1)
ax[0].clear()
ax[1].clear()
ax[0].imshow(pred.transpose(),aspect='auto')
ax[1].imshow(tgt.transpose(),aspect='auto')


## use data from speech production without slient interval ##
task_interval=[[26,100],[315,373],[540,627],[798,913],[1055,1267],[1331,1437],[1580,1671],[1818,1926],[2078,2181],[2348,2456]]
pred1=[pred[a[0]:a[1],:] for a in task_interval]
tgt1=[tgt[a[0]:a[1],:] for a in task_interval]
pred2=np.concatenate(pred1,axis=0)
tgt2=np.concatenate(tgt1,axis=0)

ax[0].clear()
ax[1].clear()
ax[0].imshow(pred2.transpose(),aspect='auto')
ax[1].imshow(tgt2.transpose(),aspect='auto')

corr2 = []  # transformer: mean=0.8677304214419497 seq2seq:0.8434792922232249
for specBin in range(pred2.shape[1]):
    r, p = pearsonr(pred2[:, specBin], tgt2[:, specBin])
    corr2.append(r)
mean_corr2=sum(corr2) / len(corr2)
print('corr:'+str(mean_corr2)) # 0.8388374582922777

from sklearn.metrics import mean_squared_error
loss2=mean_squared_error(pred2, tgt2)
print('MSE:'+str(loss2)) # 0.1391985576713361


# use all period
transformer_cc=[0.579,0.649,0.821,0.717,0.515,0.870,0.805,0.682,0.550,0.693]
transformer_cc_err=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063]
transformer_mse=[0.112,0.220,0.089,0.103,0.193,0.044,0.083,0.163,0.140,0.068]
transformer_mse_err=[0.061,0.032,0.023,0.084,0.065,0.072,0.059,0.041,0.064,0.066]

# use only the speech production intervals
transformer_cc2=[0.479,0.609,0.721,0.607,0.405,0.770,0.695,0.602,0.380,0.663]
transformer_cc_err2=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063]
transformer_mse2=[0.162,0.290,0.119,0.183,0.203,0.144,0.183,0.253,0.220,0.138]
transformer_mse_err2=[0.061,0.032,0.023,0.084,0.065,0.072,0.059,0.041,0.064,0.066]

fig,ax=plt.subplots()
ax.clear()
from pre_all import colors
sids=np.arange(1,11)
# use all period
ax.plot(sids, transformer_cc, linestyle="-", color=colors[0])
ax.plot(sids, transformer_mse, linestyle="-", color=colors[1])
# use only the speech production
ax.plot(sids, transformer_cc2, linestyle="-.", color=colors[2])
ax.plot(sids, transformer_mse2, linestyle="-.", color=colors[3])
ax.legend(['cc_all','mse_all','cc_speech','mse_speech'])

# all
ax.fill_between(sids, [transformer_cc[i-1]-transformer_cc_err[i-1] for i in sids], [transformer_cc[i-1]+transformer_cc_err[i-1] for i in sids], alpha=0.5,color=colors[0])
ax.fill_between(sids, [transformer_mse[i-1]-transformer_mse_err[i-1] for i in sids], [transformer_mse[i-1]+transformer_mse_err[i-1] for i in sids], alpha=0.5, color=colors[1])
#speech productioin
ax.fill_between(sids, [transformer_cc2[i-1]-transformer_cc_err2[i-1] for i in sids], [transformer_cc2[i-1]+transformer_cc_err2[i-1] for i in sids], alpha=0.5,color=colors[2])
ax.fill_between(sids, [transformer_mse2[i-1]-transformer_mse_err2[i-1] for i in sids], [transformer_mse2[i-1]+transformer_mse_err2[i-1] for i in sids], alpha=0.5, color=colors[3])

ax.set_xticks(np.arange(1,11))
ax.set_xticklabels(sids)

filename=result_dir+'loss_without_silence.pdf'
fig.savefig(filename)


