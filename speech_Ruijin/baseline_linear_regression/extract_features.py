import math
import os

import pandas as pd
import numpy as np
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
from scipy.io import wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
import example_speech.SingleWordProductionDutch.MelFilterBank as mel
from pre_all import computer
from speech_Ruijin.config import extra_EEG_extracting

# Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]


def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform

    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    # Linear detrend
    data = scipy.signal.detrend(data, axis=0)  # low frequency trend
    # Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70 / (sr / 2), 170 / (sr / 2)], btype='bandpass', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)  # (307511, 127)
    # Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98 / (sr / 2), 102 / (sr / 2)], btype='bandstop', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    # Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148 / (sr / 2), 152 / (sr / 2)], btype='bandstop', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    # Create feature space
    data = np.abs(hilbert3(data))
    # Number of windows
    numWindows = int(np.floor((data.shape[0] - windowLength * sr) / (frameshift * sr)))  # 30025
    feat = np.zeros((numWindows, data.shape[1]))  # (30025, 127)
    for win in range(numWindows):
        start = int(np.floor((win * frameshift) * sr))
        stop = int(np.floor(start + windowLength * sr))
        feat[win, :] = np.mean(data[start:stop, :], axis=0)
    return feat


def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors

    Parameters
    ----------
    features: array (windows, channels) # (30025, 127)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked = np.zeros(
        (features.shape[0] - (2 * modelOrder * stepSize), (2 * modelOrder + 1) * features.shape[1]))  # (29985, 1143)
    for fNum, i in enumerate(range(modelOrder * stepSize, features.shape[0] - modelOrder * stepSize)):
        ef = features[i - modelOrder * stepSize:i + modelOrder * stepSize + 1:stepSize, :]  # (9, 127)
        featStacked[fNum, :] = ef.flatten()  # Add 'F' if stacked the same as matlab
    return featStacked


def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode

    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows = int(np.floor((labels.shape[0] - windowLength * sr) / (frameshift * sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w * frameshift) * sr))
        stop = int(np.floor(start + windowLength * sr))
        newLabels[w] = scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels


def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01, melbins=23):
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
    mfb = mel.MelFilterBank(spectrogram.shape[1], melbins, sr)  # 23-80
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram


def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names

    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))), 1, 2 * modelOrder + 1).T
    for i, off in enumerate(range(-modelOrder, modelOrder + 1)):
        names[i, :] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


def dataset(dataset_name='mydata', sid=1, use_channels=False, session=1, test_shift=0, melbins=23, stacking=True, modelOrder=5,
            stepSize=5,winL=0.05, frameshift=0.01,target_SR = 16000,return_original_audio=False,use_the_official_tactron_with_waveglow=False):

    if dataset_name == 'mydata':
        from scipy.io import wavfile

        participant = 'mydate'
        from speech_Ruijin.config import data_dir
        filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session) #+ ".npy"
        if sid>=5:
            filename=filename+'_curry_crop.npy'
        else:
            filename=filename+'.npy'
        print('Loading ' + filename + '.')
        eeg = np.load(filename)
        eeg_sr = 1000
        # test_shift=int(mydate_shift*eeg_sr)
        # eeg = eeg[:, extra_EEG_extracting * eeg_sr:-extra_EEG_extracting * eeg_sr].transpose() # extra extra_EEG_extracting second from beginning and ending
        assert test_shift < extra_EEG_extracting * eeg_sr
        eeg = eeg[:,
              extra_EEG_extracting * eeg_sr + test_shift:-extra_EEG_extracting * eeg_sr + test_shift].transpose()  # extra extra_EEG_extracting second from beginning and ending

        filename = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session) + "_denoised.wav"
        print('Loading ' + filename + '.')
        audio_sr, audio = wavfile.read(filename)
        #target_SR = 16000
    elif dataset_name == 'SingleWordProductionDutch':  # not my data
        if computer == 'mac':
            path_bids = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch'
            path_output = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch/features'
        elif computer == 'workstation':
            path_bids = r'H:\Long\data\SingleWordProductionDutch-iBIDS'
            path_output = r'H:\Long\data\SingleWordProductionDutch-iBIDS\features'
        elif computer == 'Yoga':
            path_bids = r'D:\data\BaiduSyncdisk\SingleWordProductionDutch'
        elif computer == 'google':
            path_bids = r'/content/drive/MyDrive/data/SingleWordProductionDutch'
            path_output = '/content/drive/MyDrive/data/SingleWordProductionDutch/features'
        # participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
        # for p_id, participant in enumerate(participants['participant_id']):

        # Load data
        participant = 'sub-' + f"{sid:02d}"
        filename = os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_ieeg.nwb')
        print('Loading ' + filename + '.')
        io = NWBHDF5IO(filename, 'r')
        nwbfile = io.read()
        # sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]  # (307523, 127)
        if use_channels:
            eeg=eeg[:,use_channels]
        eeg_sr = 1024
        # audio
        audio = nwbfile.acquisition['Audio'].data[:]  # (14414532,)
        audio_sr = 48000
        #target_SR = 16000
        # words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)  # (307511,)
        io.close()
    elif dataset_name == 'Huashan':
        from scipy.io import wavfile
        from speech_Huashan.config import data_dir as data_dir_Huashan
        sub = '0614'
        exp = 48
        clip = 1
        filename = data_dir_Huashan + sub + '/processed/' + str(exp) + '_clip' + str(clip) + '.npy'
        print('Loading ' + filename + '.')
        eeg = np.load(filename).transpose()  # (105376, 256)
        eeg_sr = 1000
        filename = data_dir_Huashan + sub + '/processed/' + str(exp) + '_clip' + str(clip) + '.wav'
        print('Loading ' + filename + '.')
        audio_sr, audio = wavfile.read(filename)  # (4591216,)
        #target_SR = 22050  # 44100-->22050
        # audio = np.clip(audio / np.max(np.abs(audio)), -1, 1)
    # if use_the_official_tactron_with_waveglow:
    #     target_SR = 22050
    #     frameshift = 256 / target_SR
    #     winL = 1024 / target_SR
    # Extract HG features and average the window
    feat = extractHG(eeg, eeg_sr, windowLength=winL, frameshift=frameshift)  # (307523, 127)-->(30026, 127)
    # TODO: channel selection based on the correlation coefficient

    # Stack features
    if stacking:
        print('Stacking features using order: ' + str(modelOrder) + ',and step size: ' + str(stepSize) + '.')
        feat = stackFeatures(feat, modelOrder=modelOrder, stepSize=stepSize)  # (29986, 1143)
    else:
        modelOrder = 0
        stepSize = 0
        print('No features stacking.')

    # Process Audio
    import copy,resampy
    original_audio=copy.deepcopy(audio)
    audio=resampy.resample(audio, audio_sr, target_SR) # 48000 to 22050
    #audio = scipy.signal.decimate(audio, int(audio_sr / target_SR))  # (4805019,)
    audio_sr = target_SR
    if not use_the_official_tactron_with_waveglow:
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)  # 32767??
    # os.makedirs(os.path.join(path_output), exist_ok=True)
    # scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sr,scaled)

    # Extract spectrogram
    if use_the_official_tactron_with_waveglow:
        import torch
        from speech_Ruijin.utils import TacotronSTFT
        mel_bins = 23  # hop:160, win:800

        if use_the_official_tactron_with_waveglow:
            mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256,
                                           win_length=1024,
                                           n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                                           mel_fmax=8000)

        else:
            mel_transformer = TacotronSTFT(filter_length=1024, hop_length=int(frameshift * target_SR),
                                           win_length=int(winL * target_SR),
                                           n_mel_channels=mel_bins, sampling_rate=target_SR, mel_fmin=0.0,
                                           mel_fmax=target_SR // 2)
        audio = np.clip(audio / (np.max(np.abs(audio))), -1, 1)
        # (23, 30031), mean: -8.361664; max:1.0912809; min:-11.312493
        melSpec = mel_transformer.mel_spectrogram(
            torch.from_numpy(audio)[np.newaxis, :].to(torch.float32)).squeeze().numpy()
        #extra = melSpec.shape[1] - feat.shape[0]
        #remove = math.floor(extra / 2)
        #melSpec = melSpec.transpose()[remove:-remove, :]  # [3:-3,:]
        melSpec = melSpec.transpose()
    else:
        #  (30025, 23), mean:4.676236819560302, max: 14.057435370150962; min: 1.8680193789551505
        melSpec = extractMelSpecs(scaled, audio_sr, melbins=melbins, windowLength=winL, frameshift=frameshift)  #

    # Align to EEG features
    melSpec = melSpec[modelOrder * stepSize:melSpec.shape[0] - modelOrder * stepSize, :]  # (29986, 23) (25867, 80)
    # adjust length (differences might occur due to rounding in the number of windows)
    if melSpec.shape[0] != feat.shape[0]:
        print('EEG and mel difference:' + str(melSpec.shape[0] - feat.shape[0]) + '.')
        tLen = np.min([melSpec.shape[0], feat.shape[0]])
        melSpec = melSpec[:tLen, :] # (25863, 80)
        feat = feat[:tLen, :] # (25863, 127)
    else:
        print('EEG and mel lengths are the same. ')
    if return_original_audio:
        return feat, melSpec,original_audio
    else:
        return feat, melSpec


