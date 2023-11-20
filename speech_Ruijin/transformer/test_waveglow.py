import torch
from common_dl import device
from speech_Ruijin.transformer.utils import averaging

waveglow=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
#map_location=torch.device('cpu')) #not_working
#,model_dir='H:/Long/data/speech_Ruijin/models' #not_working

waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

text = "hello world, I missed you so much"
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
sequences, lengths = utils.prepare_input_sequence([text])

with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths) #mel: torch.Size([1, 80, 185])
    audio = waveglow.infer(mel) # torch.Size([1, 44800])
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050 # higher rate will speed up the audio
import numpy as np
np.save('C:/Users/wuxiaolong/我的云端硬盘/tmp/model/audio.npy',audio_numpy)

folder='H:/Long/data/speech_Ruijin/seq2seq_transformer/SingleWordProductionDutch/mel_80/sid_3/2023_08_31_08_41/'
pred=folder+'prediction.npy'
pred=np.load(pred)
tgt=folder+'truth.npy'
tgt=np.load(tgt)

pred[:, 0, :] = pred[:, 1, :]
tgt[:, 0, :] = tgt[:, 1, :]

from speech_Ruijin.transformer.opt import opt_SingleWordProductionDutch as opt
import math
frameshift = opt['frameshift']
win = math.ceil(opt['win'] / frameshift)  # int: steps

win_y=win
shift_y=stride_test = opt['stride_test']
avg_tgt = averaging(tgt, win_y, shift_y)  # (30750, 40)
avg_pred = averaging(pred, win_y, shift_y)  # (6121, 100, 40)-->(6220, 40)
avg_tgt=avg_tgt[10:-10,:] # (2970, 80)
avg_pred=avg_pred[10:-10,:] # (2970, 80)
file=folder+'avg_pred.npy'
np.save(file,avg_pred)
file=folder+'avg_tgt.npy'
np.save(file,avg_tgt)

import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1)
vmin=avg_tgt.min()
vmax=avg_tgt.max()
ax[0].imshow(avg_tgt.transpose(),aspect='auto',vmin=vmin,vmax=vmax)
ax[1].imshow(avg_pred.transpose(),aspect='auto',vmin=vmin,vmax=vmax)


#TODO: minmax the mel before waveglow.infer
from sklearn.preprocessing import MinMaxScaler
min,max=mel.cpu().min().item(),mel.cpu().max().item()
scaler = MinMaxScaler(feature_range=(min,max))
avg_tgt_norm=scaler.fit_transform(avg_tgt)
avg_pred_norm=scaler.fit_transform(avg_pred)
ax[0].imshow(avg_tgt_norm.transpose(),aspect='auto')
ax[1].imshow(avg_pred_norm.transpose(),aspect='auto')

pred=avg_pred_norm[np.newaxis,:,:].transpose(0,2,1)
tgt=avg_tgt_norm[np.newaxis, :, :].transpose(0, 2, 1)

with torch.no_grad():
    #mel, _, _ = tacotron2.infer(sequences, lengths) #mel: torch.Size([1, 80, 185])
    audio_pred = waveglow.infer(torch.from_numpy(pred).to(device).float()) # torch.Size([1, 44800])
    audio_tgt = waveglow.infer(torch.from_numpy(tgt).to(device).float())

import scipy.io.wavfile as wavfile
audio_sr=16000
wavfile.write(folder+'pred.wav', audio_sr, audio_pred.to('cpu').numpy().squeeze())
wavfile.write(folder+'tgt.wav', audio_sr, audio_tgt.to('cpu').numpy().squeeze())

audio_numpy=np.load('/Users/xiaowu/My Drive/tmp/model/audio.npy')
import sounddevice as sd
rate=48000
sd.play(audio_numpy,rate)
sd.stop()



