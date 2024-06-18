import mne
import numpy as np
import matplotlib.pyplot as plt
import librosa
from speech_Ruijin.baseline_linear_regression.extract_features import read_raw_data
from speech_Ruijin.config import result_dir

fig,axes=plt.subplots(2,1)
eeg, eeg_sr, audio, audio_sr=read_raw_data(dataset_name='mydata', sid=1, use_channels=False, session=1)
# # eeg.shape:(307511, 127)
# info = mne.create_info(ch_names=["seeg"+str(i) for i in range(127)], ch_types=["eeg"] * 127, sfreq=1024)
# raw = mne.io.RawArray(eeg.transpose(), info)
#
# fMin,fMax=1,500
# fstep=1
# freqs=np.arange(fMin,fMax,fstep)
# tfr=raw.compute_tfr(method='morlet',freqs=freqs,picks=[1,],tmin=1,tmax=20)
#
#
# axes[0].imshow(tfr.data[0,:,:],aspect='auto')
# axes[0].set_yscale('symlog')
# axes[0].clear()
#
# # audio.shape:(14414532,) 48000
# audio2=mne.filter.resample(audio,down=24)
# info2 = mne.create_info(ch_names=["audio",], ch_types=["eeg",] , sfreq=2000)
# raw2 = mne.io.RawArray(audio2[np.newaxis,:], info2)
#
# fMin,fMax=1,1000
# fstep=1
# freqs2=np.arange(fMin,fMax,fstep)
# tfr2=raw2.compute_tfr(method='morlet',freqs=freqs2,picks=[0,],tmin=1,tmax=20)
# axes[1].imshow(tfr2.data[0,:,:],aspect='auto')
# axes[1].set_yscale('symlog')
# axes[1].clear()

#D = np.abs(librosa.stft(audio[1*48000:20*48000]))**2
#dd=librosa.feature.melspectrogram(y=audio2[1*2000:20*2000],sr=2000)
dd=librosa.feature.melspectrogram(y=audio[1*48000:20*48000],sr=48000,n_mels=128,fmax=8000)
S_dB = librosa.power_to_db(dd, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=48000,fmax=8000, ax=axes[0])
c1=fig.colorbar(img,ax=axes[0], format="%+2.0f dB")
c1.remove()

eeg1=librosa.feature.melspectrogram(y=eeg[1*1024:20*1024,0].squeeze(),sr=1024,n_mels=128,fmax=500)
S_dB2 = librosa.power_to_db(eeg1, ref=np.max)
img2 = librosa.display.specshow(S_dB2, x_axis='time', y_axis='mel', sr=1024,fmax=500, ax=axes[1])
c2=fig.colorbar(img2,ax=axes[1], format="%+2.0f dB")

filename=result_dir+'spectrogram_eeg_VS_audio.pdf'
fig.savefig(filename)

