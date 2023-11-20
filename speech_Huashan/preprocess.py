import os

import h5py
import mne
import numpy as np
from scipy.io import wavfile
from speech_Huashan.config import data_dir
sf_EEG=4000
eeg_file=['/Volumes/Samsung_T5/data/speech_Huashan/0614/processed/subjects_3.exps_47.mat',
          '/Volumes/Samsung_T5/data/speech_Huashan/0614/processed/subjects_3.exps_48.mat'
          ]
audio_file=['/Volumes/Samsung_T5/data/speech_Huashan/0614/processed/subjects_3.exps_47_denoised.wav',
            '/Volumes/Samsung_T5/data/speech_Huashan/0614/processed/subjects_3.exps_48_denoised.wav',
            ]
# exp47: 2m2.720s(122.720)--3m48.095(228.095); 4m9.251(249.251)--6m54.267(414.267)
# exp48: 12.628s(12628)--4m59.579(299579)
clip47=[[122720,228095],[249251,414267]]
clip48=[[12628,299579],]
### exp 47
filei=0
mat=h5py.File(eeg_file[filei],'r')
data=mat['d']['frame'] #(256, 1545600)
ttl=mat['d']['ttl'] #(1, 1545600)
len_EEG=data.shape[1]/sf_EEG # 1: 431.904s;
chn_names=["ecog"]*data.shape[0]
chn_types=["ecog"]*data.shape[0]
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=sf_EEG)
raw = mne.io.RawArray(data, info)
raw.resample(sfreq=1000)
raw.load_data().notch_filter(np.arange(50, 451, 50))
raw47_1=raw.copy().crop(tmin=122.720, tmax=228.095) # 105376 ms
## filter the audio EEG
raw47_1.plot()
raw47_1.filter(l_freq=5,h_freq=None)
raw47_2=raw.copy().crop(tmin=249.251, tmax=414.267) # 165017 ms
raw.filter(l_freq=5,h_freq=None)
raw.plot()
filename=data_dir+'0614/processed/47_clip1.npy'
np.save(filename,raw47_1.get_data())
filename=data_dir+'0614/processed/47_clip2.npy'
np.save(filename,raw47_2.get_data())

sf_audio, audio = wavfile.read(audio_file[filei])
import resampy
target_sf=16000
audio = resampy.resample(audio, sf_audio, target_sf) # (6910258,)
audio_clip1=audio[int(122.720*target_sf):int(228.095*target_sf)]
audio_clip2=audio[int(249.251*target_sf):int(414.267*target_sf)]
filename=data_dir+'0614/processed/47_clip1.wav'
wavfile.write(filename,target_sf,audio_clip1)
filename=data_dir+'0614/processed/47_clip2.wav'
wavfile.write(filename,target_sf,audio_clip2)

### exp 48
filei=1
mat=h5py.File(eeg_file[filei],'r')
data=mat['d']['frame'] #(256, 1545600)
ttl=mat['d']['ttl'] #(1, 1545600)
len_EEG=data.shape[1]/sf_EEG # 1: 431.904s;
chn_names=["ecog"]*data.shape[0]
chn_types=["ecog"]*data.shape[0]
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=sf_EEG)
raw = mne.io.RawArray(data, info)
raw.resample(sfreq=1000)
raw.load_data().notch_filter(np.arange(50, 451, 50))
raw48_1=raw.copy().crop(tmin=12.628, tmax=299.579) #
filename=data_dir+'0614/processed/48_clip1.npy'
np.save(filename,raw48_1.get_data())

sf_audio, audio = wavfile.read(audio_file[filei])
import resampy
target_sf=16000
audio = resampy.resample(audio, sf_audio, target_sf) # (6910258,)
audio_clip1=audio[int(12.628*target_sf):int(299.579*target_sf)]
filename=data_dir+'0614/processed/48_clip1.wav'
wavfile.write(filename,target_sf,audio_clip1)

raw.filter(l_freq=5,h_freq=None)
raw.plot()

sf_audio, audio = wavfile.read(audio_file[filei])
len_audio=audio.shape[0]/sf_audio # 1:431.89115646258506
# 386.47292517006804 s
error=len_EEG-len_audio # 1:12.8ms
print(error) # 0.031895691609975074, 0.0128435374149376, 0.04194104308390045

