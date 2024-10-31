import numpy as np
from scipy.io import wavfile
import mne
from scipy.stats import spearmanr

from speech_Dutch.config import data_dir
import torch
from speech_Dutch.utils import TacotronSTFT

sid=3
session=0
sf_EEG=1000

filename = data_dir + "P" + str(sid) + "/raw/EEG/session" + str(session + 1) + ".npy"
EEG = np.load(filename)
EEG=EEG[:,1*sf_EEG:-1*sf_EEG] # (118, 556270)
# audio data
filename = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session + 1) + "_denoised.wav"
sf_audio, audio = wavfile.read(filename)

ratio=EEG.shape[1]/audio.shape[0]
ch_types = ['eeg']*EEG.shape[0]
ch_names = ['eeg']*EEG.shape[0]
info = mne.create_info(ch_names=ch_names, sfreq=sf_EEG, ch_types=ch_types)
raw = mne.io.RawArray(EEG, info)
data=raw.filter(70,170).get_data() # (118, 558270)

tmp = np.clip(audio / np.max(np.abs(audio)),-1,1)
mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024,n_mel_channels=80, sampling_rate=48000, mel_fmin=0.0, mel_fmax=8000.0)
mel = mel_transformer.mel_spectrogram(torch.from_numpy(tmp)[None,:].to(torch.float32)).numpy().squeeze() # (80, 104301)
time_index_mel=list(range(mel.shape[1]))
time_index_EEG=[int(i*ratio) for i in time_index_mel]

x=EEG[:,time_index_EEG]
y=mel[:,time_index_mel] # (80, 104301)
y_mean=np.mean(y, axis=0) # (104301,)

ch_num=x.shape[0]
cc=[0]*ch_num
ps=[0]*ch_num
for c in range(ch_num):
    xi=x[c,:]
    cc[c], ps[c] = spearmanr(xi, y_mean)
max(cc)

import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax.plot(x[0,:],y_mean)
