import sys,os,glob
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

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pre_all import computer, device, running_from_CMD
from common_dl import myDataset
from speech_Dutch.baseline_linear_regression.reconstruction_minimal import createAudio
from speech_Dutch.config import data_dir
import numpy as np
import torch.nn as nn
import numpy.random as npr
from speech_Dutch.transformer.lib.train import make_model #*
# if error happens, change this: "from scipy.misc import logsumexp" to "from scipy.special import logsumexp"
import torch
from torch.utils.data import DataLoader
from speech_Dutch.transformer.utils import test_transformer_model, averaging
from speech_Dutch.transformer.opt import opt_transformer
import math
from speech_Dutch.baseline_linear_regression.extract_features import dataset
from speech_Dutch.utils import fold_2d23d
fig, ax = plt.subplots(2, 1)
npr.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)

model_name = 'seq2seq_transformer'  # 'seq2seq_transformer'/'seq2seq'
if running_from_CMD:
    sid = int(float(sys.argv[1]))
    dataname = sys.argv[2] #'SingleWordProductionDutch'
    time_stamp=sys.argv[3]
    mel_bins = sys.argv[4]
else:
    dataname='SingleWordProductionDutch'
    sid=3
    time_stamp='dummy'
    mel_bins = 80

if dataname=='mydata':#'SingleWordProductionDutch'/'mydata' # 'SingleWordProductionDutch' (10 subjects)/'stereoEEG2speech_master'(3 subjects)
    from speech_Dutch.transformer.opt import opt_mydata as opt
elif dataname=='SingleWordProductionDutch':
    from speech_Dutch.transformer.opt import opt_SingleWordProductionDutch as opt

############
window_eeg=opt['window_eeg']
target_SR=opt['target_SR']
use_the_official_tactron_with_waveglow = opt['use_the_official_tactron_with_waveglow']
winL = opt['winL']
frameshift = opt['frameshift']
sf_EEG = opt['sf_EEG']
stride = opt['stride']
step_size = opt['step_size']
model_order = opt['model_order']
win = math.ceil(opt['win'] / frameshift)  # int: steps
history = math.ceil(opt['history'] / frameshift)  # int(opt['history']*sf_EEG) # int: steps
baseline_method = opt['baseline_method']
stride_test = opt['stride_test']
lr=opt_transformer['lr']
norm_EEG=opt['norm_EEG']#True
norm_mel=opt['norm_mel'] #False
##################
print('baseline_method: ' + str(opt['baseline_method']) +'; win: ' + str(win) + '; history:' +
      str(history) + '; stride:' + str(stride)+'; winL:'+str(winL)+'; frameshift: '+str(frameshift)+'.')

# device=torch.device('cpu')
if computer == 'workstation' or computer == 'Yoga':
    result_dir = data_dir + 'seq2seq_transformer/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(sid) + '/' + time_stamp + '/'
elif computer == 'mac':
    result_dir = '/Users/xiaowu/tmp/models/seq2seq_transformer/' + time_stamp + '/'

#for sid,folder_date in zip([sids[i-1] for i in sid_idx],[folder_dates[i-1] for i in sid_idx]):
#for sid,folder_date, epoch in zip([sids[i-1] for i in sid_idx],[folder_dates[i-1] for i in sid_idx], [epochs[i-1] for i in sid_idx]):
#for aaa in [1]: # no loop anymore
#folder_date = time_stamp
print('sid: '+str(sid)+'.')
if dataname == 'mydata':
    test_shift = 300
    from speech_Dutch.dataset import dataset_Dutch

    # x1, y1 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=1,test_shift=test_shift, mel_bins=mel_bins, continous=continous_data)
    # x2, y2 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=2,test_shift=300, mel_bins=mel_bins, continous=continous_data)
    x1, y1 = dataset(dataset_name=dataname, sid=sid, session=1, test_shift=0, melbins=mel_bins, stacking=False,
                     winL=winL, frameshift=frameshift)
    x2, y2 = dataset(dataset_name=dataname, sid=sid, session=2, test_shift=0, melbins=mel_bins, stacking=False,
                     winL=winL, frameshift=frameshift)
    if False:
        mu = np.mean(x1, axis=0)
        std = np.std(x1, axis=0)
        x1 = (x1 - mu) / std
        mu = np.mean(x2, axis=0)
        std = np.std(x2, axis=0)
        x2 = (x2 - mu) / std

        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
    else:
        x = x1
        y = y1
elif dataname == 'SingleWordProductionDutch':
    # x, y = get_data(dataname=dataname, sid=sid,continous_data=continous_data,mel_bins=mel_bins)  # x: (512482,127), y:(344858, 80)
    x, y = dataset(dataset_name=dataname, sid=sid, melbins=mel_bins, stacking=False, winL=winL,  target_SR =target_SR,frameshift=frameshift
                   ,use_the_official_tactron_with_waveglow=use_the_official_tactron_with_waveglow,window_eeg=window_eeg)

xy_ratio = x.shape[0]/y.shape[0]

if norm_mel:
    mu = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    y = (y - mu) / std

lenx = x.shape[0]
leny = y.shape[0]
train_x = x[:int(lenx * 0.8), :]
val_x = x[int(lenx * 0.8):int(lenx * 0.9), :]
test_x = x[int(lenx * 0.9):, :]
train_y = y[:int(leny * 0.8), :]
val_y = y[int(leny * 0.8):int(leny * 0.9), :]
test_y = y[int(leny * 0.9):, :] #(3003, 80)

#norm_EEG = opt['norm_EEG'] #False
if norm_EEG:
    mu = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    train_x = (train_x - mu) / std
    val_x = (val_x - mu) / std
    test_x = (test_x - mu) / std

win_x, win_y, shift_x, shift_y =  (win+history)*xy_ratio, win,stride*xy_ratio,stride
# win_x, win_y, shift_x, shift_y = win, win* xy_ratio, stride, stride * xy_ratio
x_test, y_test = fold_2d23d(test_x.transpose(), test_y[history:,:].transpose(), win_x, win_y, shift_x, shift_y)
x_test, y_test = x_test.transpose(0,2,1), y_test.transpose(0,2,1) #0, 2, 1
# Get input_d, output_d, timesteps from the initial dataset
input_d, output_d = x_test.shape[2], y_test.shape[2]
out_len = y_test.shape[1] #(batch,feature,time)-->(batch,time, feature)
# encoder need: (time, batch, feature)
dataset_test = myDataset(x_test, y_test)
batch_size = opt_transformer['batch_size']
dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

path_file = result_dir +  'best_model_epoch*'+'.pth' # '19_final.pth'
path_file = os.path.normpath(glob.glob(path_file)[0])

opt_transformer['src_d'] = input_d  # input dimension
opt_transformer['tgt_d'] = output_d  # output dimension
opt_transformer['out_len'] = out_len

encoder_only = False
spatial_attention = False
model = make_model(opt_transformer['src_d'], opt_transformer['tgt_d'], N=opt_transformer['Transformer-layers'],
                   d_model=opt_transformer['Model-dimensions'], d_ff=opt_transformer['feedford-size'], h=opt_transformer['headers'],
                   dropout=opt_transformer['dropout'], norm_mel=opt['norm_mel'], encoder_only=encoder_only,
                   spatial_attention=spatial_attention).to(device)
model = model.to(device)
checkpoint = torch.load(path_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
pred, tgt = test_transformer_model(sid, model, opt_transformer, dataloader_test, result_dir)
filename=result_dir+'/prediction.npy'
np.save(filename,pred)
filename = result_dir + '/truth.npy'
np.save(filename, tgt)
# or load prediction
'''
the_date = '2023_07_13_17_26'
if computer == 'mac':
    filename_pred = '/Users/xiaowu/tmp/untitled folder/prediction.npy'
    filename_tgt = '/Users/xiaowu/tmp/untitled folder/truth.npy'
elif computer == 'workstation':
    filename_pred = 'H:/Long/data/speech_Ruijin/seq2seq_transformer/' + the_date + '/best_model_epoch53_predictions.npy'
    filename_tgt = 'H:/Long/data/speech_Ruijin/seq2seq_transformer/' + the_date + '/best_model_epoch53_predictions.npy'
pred = np.load(filename_pred)  # (3066, 100, 40)
tgt = np.load(filename_tgt)  # (3066, 100, 40)
'''

#pred[:, 0, :] = pred[:, 1, :]
pred = pred[:, 1:, :]

avg_tgt = averaging(tgt, win_y, shift_y)  # (30750, 40)
avg_pred = averaging(pred, win_y, shift_y)  # (6121, 100, 40)-->(6220, 40)

del model
start = 0
#ax[0].imshow(avg_tgt.transpose(), cmap='RdBu_r', aspect='auto')
#ax[1].imshow(avg_pred.transpose(), cmap='RdBu_r', aspect='auto')
avg_tgt=avg_tgt[10:-10,:]
avg_pred=avg_pred[10:-10,:]
vmin=avg_tgt.min()
vmax=avg_tgt.max()
ax[0].imshow(avg_tgt.transpose(),aspect='auto',vmin=vmin,vmax=vmax)
ax[1].imshow(avg_pred.transpose(),aspect='auto',vmin=vmin,vmax=vmax)
figname=result_dir+'plot.png'
fig.savefig(figname)
ax[0].clear()
ax[1].clear()

filename = result_dir + 'mel_pred.npy'
np.save(filename, avg_pred)
filename = result_dir + 'mel_tgt.npy'
np.save(filename, avg_tgt)
#avg_tgt=avg_tgt.transpose()
#avg_pred=avg_pred.transpose()
partial_teste = True  # only test partial testing data in the original paper
if partial_teste:
    test_on_shorter_range = 3000  # mean=0.82
else:
    test_on_shorter_range = avg_pred.shape[0]
corr = []  # transformer: mean=0.8677304214419497 seq2seq:0.8434792922232249
for specBin in range(avg_pred.shape[1]):
    r, p = pearsonr(avg_pred[:test_on_shorter_range, specBin], avg_tgt[:test_on_shorter_range, specBin])
    corr.append(r)
mean_corr=sum(corr) / len(corr)
print('corr:'+str(mean_corr))

criterion = nn.MSELoss()
loss = criterion(torch.from_numpy(avg_pred),torch.from_numpy(avg_tgt))  # 40:10.6792; 80: tensor(18.4728, dtype=torch.float64)
from sklearn.metrics import mean_squared_error
loss=mean_squared_error(avg_pred, avg_tgt)
print('MSE:'+str(loss))

filename = result_dir + 'result.txt'
with open(filename, 'w') as f:
    f.write(str(mean_corr))
    f.write('\n')
    f.write(str(loss.item()))

gen_wave=False
if gen_wave:
    # Synthesize waveform from spectrogram using Griffin-Lim
    print("Synthesizing waveform using Griffin-Lim...")
    audiosr = 10240
    winLength = 1024 / 10240
    frameshift = 10 / 10240
    import scipy.io.wavfile as wavfile
    avg_tgt2 = createAudio(avg_tgt, audiosr=audiosr, winLength=winLength, frameshift=frameshift, log=True)
    avg_pred2 = createAudio(avg_pred, audiosr=audiosr, winLength=winLength, frameshift=frameshift, log=True)
    wavfile.write(result_dir+'tgt.wav', int(audiosr), avg_tgt2)
    wavfile.write(result_dir+'pred.wav', int(audiosr), avg_pred2)


'''
# Synthesize using Nvidia WaveGlow. ( It needs 80 frequency bins)
import torch
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
#waveglow = waveglow.to('cpu')
waveglow.eval()
with torch.no_grad():
    audio_tgt = waveglow.infer(torch.from_numpy(avg_tgt.transpose())[None,:,:].to(dtype=torch.float).to('cpu'))
    audio_pred = waveglow.infer(torch.from_numpy(avg_pred.transpose())[None,:,:].to(dtype=torch.float).to('cpu'))

audio_numpy_tgt = audio_tgt[0].data.cpu().numpy()
audio_numpy_pred = audio_pred[0].data.cpu().numpy()
rate = 22050 # higher rate will speed up the audio
np.save('C:/Users/wuxiaolong/我的云端硬盘/tmp/model/audio_numpy_tgt.npy',audio_numpy_tgt)
np.save('C:/Users/wuxiaolong/我的云端硬盘/tmp/model/audio_numpy_pred.npy',audio_numpy_pred)

import sounddevice as sd
fs=10240
sd.play(avg_tgt2,fs)
sd.play(avg_pred2,fs)
sd.stop()
'''