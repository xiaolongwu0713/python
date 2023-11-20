import math
import numpy as np
import torch
from speech_Ruijin.stft import STFT
from librosa.filters import mel as librosa_mel_fn
from speech_Ruijin.audio_processing import dynamic_range_compression
from speech_Ruijin.audio_processing import dynamic_range_decompression
from speech_Ruijin.config import extra_EEG_pairring,sf_EEG
from librosa.core import resample

model=None #seq2seq/OLS/DenseModel

# fold (x,y) pair from 2D data to 3D
#data.shape:(118channel, 1080time_steps)
def fold_2d23d(x,y,wind_x,wind_y,stride_x,stride_y,):
    windows=math.floor((x.shape[1]-wind_x)/stride_x)
    extra=True if (wind_x+windows*stride_x < (x.shape[1])) else False

    #windows=math.floor(x.shape[1]/wind_x)
    #extra=True if windows<(x.shape[1]/wind_x) else False
    tmp1 = []
    tmp2 = []

    for i in range(windows):
        tmp1.append(x[:,int(i*stride_x):int(i*stride_x+wind_x)])
        tmp2.append(y[:,int(i*stride_y):int(i*stride_y+wind_y)])
        #tmp1.append(x[:, i * wind_x:(i + 1) * wind_x])
        #tmp2.append(y[:, i * wind_y:(i + 1) * wind_y])
    if extra:
        tmp1.append(x[:, -int(wind_x):])
        tmp2.append(y[:, -int(wind_y):])
    # in case length is not the same
    shortest=min([i.shape[1] for i in tmp2])
    tmp22=[j[:,:shortest] for j in tmp2]
    return (np.asarray(tmp1),np.asarray(tmp22))



def fold_2d23d2(x,y,wind_x,wind_y,stride_x,stride_y,pre_EEG,post_EEG):
    wind_x2=wind_x+pre_EEG+post_EEG
    windows=math.floor((x.shape[1]-2*extra_EEG_pairring*sf_EEG-wind_x)/stride_x)
    extra=True if (wind_x+windows*stride_x < (x.shape[1]-2*extra_EEG_pairring*sf_EEG)) else False

    #windows=math.floor(x.shape[1]/wind_x)
    #extra=True if windows<(x.shape[1]/wind_x) else False
    tmp1 = []
    tmp2 = []

    for i in range(windows):
        tmp1.append(x[:,int(i*stride_x+extra_EEG_pairring*sf_EEG-pre_EEG):int(i*stride_x+wind_x+extra_EEG_pairring*sf_EEG+post_EEG)])
        tmp2.append(y[:,i*stride_y:(i*stride_y+wind_y)])
        #tmp1.append(x[:, i * wind_x:(i + 1) * wind_x])
        #tmp2.append(y[:, i * wind_y:(i + 1) * wind_y])
    if extra:
        tmp1.append(x[:, -int(wind_x+extra_EEG_pairring*sf_EEG+pre_EEG):-int(extra_EEG_pairring*sf_EEG-post_EEG)])
        tmp2.append(y[:, -wind_y:])
    return (tmp1,tmp2)


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=48000, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length) # hop and window length are in samples.
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)    ### filter_length = number of FFT components

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
    # magnitudes: torch.Size([1, 80, 104301])
    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output # shape: torch.Size([1, 80, 104301])

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1], shape: torch.Size([1, 2026])
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        if (model=='OLS' or model=='DenseModel'):
            return mel_output[:,:,3].unsqueeze(-1)
            #stft_fn.transform pads sequence with reflection to be twice the original size.
            #hence 5 MFCC framea are produced for the 50ms window. We take the middle one which should correspond best to the original frame.
        else:
            return mel_output


