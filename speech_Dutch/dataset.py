import librosa
import numpy as np
import textgrid
import scipy
from scipy.io import wavfile

from speech_Dutch.baseline_linear_regression.extract_features import hilbert3, dataset
from speech_Dutch.config import *
from speech_Dutch.utils import TacotronSTFT
import math
from pre_all import computer
from speech_Dutch.config import data_dir


import example_speech.SingleWordProductionDutch.MelFilterBank as mel
def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms

    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows = int(np.floor((audio.shape[0] - windowLength * sr) / (frameshift * sr)))
    win = scipy.hanning(np.floor(windowLength * sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength * sr / 2 + 1))), dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w * frameshift) * sr))
        stop_audio = int(np.floor(start_audio + windowLength * sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win * a)
        spectrogram[w, :] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram


# read data
sids=[1,2]
sessions=2 # two sessions
sf_EEG=1000 # Hz
sf_audio=48000 # Hz
extra_time=1 #s
# wind_size=200 ms
def dataset_Dutch(dataset_name='SingleWordProductionDutch',baseline_method=False, sid=1, session=None, test_shift=0, wind_size=200,stride=200,mel_bins=40,\
                  val_len=4000,test_len=4000, continous=False,stepping=False):
    from pynwb import NWBHDF5IO
    import scipy

    x_train=[]
    y_train=[]

    x_session=[]
    y_session=[]

    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    target_SR = 16000
    if dataset_name=='SingleWordProductionDutch':

        if not baseline_method:
            if computer=='mac':
                path_bids = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch'
            elif computer=='workstation':
                path_bids = 'H:/Long/data/SingleWordProductionDutch-iBIDS/'
            elif computer=='google':
                path_bids='/content/drive/MyDrive/data/SingleWordProductionDutch/'
            participant='sub-'+f"{sid:02d}" #str(sid)
            #print('haha')
            filename=path_bids+participant+'/ieeg/'+participant+'_task-wordProduction_ieeg.nwb'
            io = NWBHDF5IO(filename, 'r')
            print('Read EEG file: '+filename)
            nwbfile = io.read()
            # sEEG
            EEG = nwbfile.acquisition['iEEG'].data[:] #.transpose()  # (307523, 127) 300.3154296875s
            sf_EEG = 1024

            # adding feature engineering
            # detrend
            EEG = scipy.signal.detrend(EEG, axis=0)
            # Filter High-Gamma Band
            sos = scipy.signal.iirfilter(4, [70 / (sf_EEG / 2), 170 / (sf_EEG / 2)], btype='bandpass', output='sos')
            EEG = scipy.signal.sosfiltfilt(sos, EEG, axis=0)  # (307511, 127)
            # Attenuate first harmonic of line noise
            sos = scipy.signal.iirfilter(4, [98 / (sf_EEG / 2), 102 / (sf_EEG / 2)], btype='bandstop', output='sos')
            EEG = scipy.signal.sosfiltfilt(sos, EEG, axis=0)
            # Attenuate second harmonic of line noise
            sos = scipy.signal.iirfilter(4, [148 / (sf_EEG / 2), 152 / (sf_EEG / 2)], btype='bandstop', output='sos')
            EEG = scipy.signal.sosfiltfilt(sos, EEG, axis=0)
            EEG = np.abs(hilbert3(EEG)) # TODO: test this
            EEG=EEG.transpose()

            # audio
            audio = nwbfile.acquisition['Audio'].data[:]
            audio_sr = 48000
            ratio=10
            target_SR=ratio*sf_EEG
            io.close()
            #audio = scipy.signal.decimate(audio, int(audio_sr / target_SR)) # 4805019
            import resampy
            audio = resampy.resample(audio, audio_sr, target_SR)
            mel_transformer = TacotronSTFT(filter_length=1024, hop_length=ratio, win_length=1024,
                                           n_mel_channels=mel_bins, sampling_rate=target_SR, mel_fmin=0.0, mel_fmax=target_SR//2) # mel_fmax=target_SR//2
            if continous:
                audio = np.clip(audio / (np.max(np.abs(audio))), -1, 1)
                mel = mel_transformer.mel_spectrogram(torch.from_numpy(audio)[np.newaxis, :].to(
                    torch.float32)).squeeze().numpy()  # shape:([80, 104301]);
                # in case lengths are differnt
                len_min=min(EEG.shape[1],mel.shape[1])
                return EEG[:,:len_min].transpose(), mel[:,:len_min].transpose()
            elif stepping:
                pass
        else: # baseline method
            mydata=False
            pt = 'sub-' + f"{sid:02d}"  # str(sid)
            EEG, mel = dataset(dataset_name=mydata,stacking=False,pt=pt)
            return EEG, mel # (30025, 127), (30025, 23)
    elif dataset_name=='mydata':
        EEG, mel = dataset(dataset_name=dataset_name, test_shift=test_shift, stacking=False, sid=sid,session=session)
        return EEG, mel  # (30025, 127), (30025, 23)
    elif dataset_name=='stereoEEG2speech_master':
        if computer=='mac':
            data_dir2='/Volumes/Samsung_T5/data/stereoEEG2speech_master/'
        elif computer == 'workstation':
            data_dir2='H:/Long/data/stereoEEG2speech_master/'
        EEG=np.load(data_dir2+'p'+str(sid)+'_sEEG_gamma.npy')
        audio=np.load(data_dir2+'p'+str(sid)+'_audio_final.npy')
        sf_EEG = 1024
        sf_audio=22050

        audio = scipy.signal.decimate(audio, int(sf_audio / target_SR))
        sf_audio = target_SR
        mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024,
                                       n_mel_channels=mel_bins, sampling_rate=sf_audio, mel_fmin=0.0, mel_fmax=8000.0)
        if continous:
            audio = np.clip(audio / (np.max(np.abs(audio))), -1, 1)
            mel = mel_transformer.mel_spectrogram(torch.from_numpy(audio)[np.newaxis, :].to(
                torch.float32)).squeeze().numpy()  # shape:([80, 104301]);
            return EEG, mel

    EEG_train=EEG[:,:-(val_len+test_len)] # (118, 548270)
    EEG_val=EEG[:,-(val_len+test_len):-test_len] # (118, 4000)
    EEG_test=EEG[:,-test_len:] # (118, 4000)
    audio_train=audio[:-int((val_len+test_len)/sf_EEG*sf_audio)] # (26316960,)
    audio_val=audio[-int((val_len+test_len)/sf_EEG*sf_audio):-int(test_len/sf_EEG*sf_audio)] # (192000,)
    audio_test=audio[-int(test_len/sf_EEG*sf_audio):] # (192000,)

    EEG2=EEG_train
    audio2=audio_train

    #(80, 104301)
    audio_train=np.clip(audio_train / (np.max(np.abs(audio_train))),-1,1)
    mel2 = mel_transformer.mel_spectrogram(torch.from_numpy(audio_train)[np.newaxis, :].to(torch.float32)).squeeze().numpy()  # shape:([80, 104301]);
    ratio=EEG_train.shape[1]/mel2.shape[1]
    steps=math.floor((EEG_train.shape[1]-wind_size)/stride)
    for step in range(steps):
        stop_at=step*stride+wind_size
        x_session.append(EEG_train[:,step*stride:stop_at]) # (118, 200)
        stop_at2=int(stop_at/ratio)
        y_session.append(mel2[:,stop_at2])
    x_train.append(np.asarray(x_session))
    y_train.append(np.asarray(y_session))
    x_train=np.concatenate(x_train,axis=0) # (trials,channel,time) (5181, 118, 200)
    y_train=np.concatenate(y_train,axis=0) # (trials,mel_bins)  (5181, 80)
    # (80, 751)
    audio_val=np.clip(audio_val / (np.max(np.abs(audio_val))),-1,1)
    mel_val=mel_transformer.mel_spectrogram(torch.from_numpy(audio_val)[np.newaxis, :].to(torch.float32)).squeeze().numpy()
    audio_test =np.clip(audio_test / (np.max(np.abs(audio_test))),-1,1)
    mel_test=mel_transformer.mel_spectrogram(torch.from_numpy(audio_test)[np.newaxis, :].to(torch.float32)).squeeze().numpy()
    x_val=[]
    y_val=[]
    for i in range(mel_val.shape[1]):
        stop=-int(i*ratio)
        start=-int(i*ratio+wind_size)
        if stop==0:
            x_val.append(EEG_val[:, start:])
            y_val.append(mel_val[:, -(i + 1)])
        elif EEG_val.shape[1]+stop>wind_size:
            x_val.append(EEG_val[:,start:stop])
            y_val.append(mel_val[:, -(i + 1)])

    x_test = []
    y_test = []
    for i in range(mel_test.shape[1]):
        stop = -int(i * ratio)
        start = -int(i * ratio + wind_size)
        if stop == 0:
            x_test.append(EEG_test[:, start:])
            y_test.append(mel_test[:, -(i + 1)])
        elif EEG_test.shape[1] + stop > wind_size:
            x_test.append(EEG_test[:, start:stop])
            y_test.append(mel_test[:, -(i + 1)])

    train_ds = myDataset(x_train, y_train,norm=True)
    val_ds = myDataset(x_val, y_val,norm=True)
    test_ds = myDataset(x_test, y_test,norm=True)

    return train_ds,val_ds,test_ds





