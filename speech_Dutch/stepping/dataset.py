import math

import numpy as np
import torch
from textgrid import textgrid
from common_dl import myDataset
from speech_Dutch.baseline_linear_regression.extract_features import dataset
from speech_Dutch.config import data_dir, extra_EEG_extracting
from speech_Dutch.utils import TacotronSTFT

# dataname: 'mydata'/'SingleWordProductionDutch'
def dataset_stepping(dataname='mydata', sid=1,session=None, wind_size=200,
                     stride=1,mel_bins=23,val_len=4000,test_len=4000,shift=None,test_only=False):
    if dataname=='SingleWordProductionDutch':
        #pt = 'sub-' + f"{sid:02d}"  # str(sid)
        EEG, mel = dataset(dataset_name=dataname, sid=sid, stacking=False,winL=0.05,frameshift=0.001,melbins=mel_bins)
        # train/val/test split
        trainL=int(EEG.shape[0]*0.7)
        valL=int(EEG.shape[0]*0.9)
        EEG_train=EEG[:trainL,:]
        mel_train=mel[:trainL,:]
        EEG_val=EEG[trainL:valL,:]
        mel_val=mel[trainL:valL,:]
        EEG_test=EEG[valL:,:]
        mel_test=mel[valL:,:]

        x_test = []
        y_test = []
        steps = math.floor((EEG_test.shape[0] - wind_size) / stride)
        for i in range(steps):
            start = i * stride
            stop = start + wind_size
            between = int((start + stop) / 2)
            x_test.append(EEG_test[start:stop, :])
            y_test.append(mel_test[between])
        x_test = np.asarray(x_test).transpose(0, 2, 1)
        y_test = np.asarray(y_test)
        test_ds = myDataset(x_test, y_test, norm=True)
        if test_only:
            return test_ds

        x_train = []
        y_train = []
        steps = math.floor((EEG_train.shape[0] - wind_size) / stride)
        for i in range(steps):
            start = i * stride
            stop = start + wind_size
            between=int((start+stop)/2)
            x_train.append(EEG_train[start:stop, :])
            y_train.append(mel_train[between])
        x_train = np.asarray(x_train).transpose(0,2,1)
        y_train = np.asarray(y_train)

        x_val = []
        y_val = []
        steps = math.floor((EEG_val.shape[0] - wind_size) / stride)
        for i in range(steps):
            start=i*stride
            stop=start+wind_size
            between = int((start + stop) / 2)
            x_val.append(EEG_val[start:stop,:])
            y_val.append(mel_val[between])
        x_val = np.asarray(x_val).transpose(0,2,1)
        y_val = np.asarray(y_val)


        train_ds = myDataset(x_train, y_train, norm=True)
        val_ds = myDataset(x_val, y_val, norm=True)


        return train_ds, val_ds, test_ds

    x_train=[]
    y_train=[]
    for session in range(sessions):
        x_session=[]
        y_session=[]
        # EEG data
        if highgamma:
            filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session + 1) + "_highgamma.npy"
        else:
            filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session + 1) + ".npy"
        EEG = np.load(filename)
        # audio data
        filename = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session + 1) + "_denoised.wav"
        sf_audio, audio = wavfile.read(filename)
        target_SR = 16000
        audio = scipy.signal.decimate(audio, int(audio_sr / target_SR))  # (8900320,)
        sf_audio=target_SR
        # speech intervals
        gridfile = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session + 1) + "_denoised.TextGrid"
        tg = textgrid.TextGrid.fromFile(gridfile)
        intervals = []
        intervals_bad = []
        for it in tg[0]:
            if it.mark == 'speech':
                tmp = [it.minTime, it.maxTime]
                intervals.append(tmp)
            if it.mark == 'bad':
                tmp = [it.minTime, it.maxTime]
                intervals_bad.append(tmp)
        # extra 1 second EEG data before and after audio
        assert extra_EEG_extracting * 2 == EEG.shape[1] / sf_EEG - audio.shape[0] / sf_audio
        if shift is not None:
            EEG=EEG[:,int((extra_EEG_extracting+shift)*sf_EEG):-int((extra_EEG_extracting-shift)*sf_EEG)]
        else:
            EEG=EEG[:,int(extra_EEG_extracting)*sf_EEG:-int(extra_EEG_extracting)*sf_EEG] # (118, 556270)

        if session==0:
            EEG_train=EEG[:,:-(val_len+test_len)] # (118, 548270)
            EEG_val=EEG[:,-(val_len+test_len):-test_len] # (118, 4000)
            EEG_test=EEG[:,-test_len:] # (118, 4000)
            audio_train=audio[:-int((val_len+test_len)/sf_EEG*sf_audio)] # (26316960,)
            audio_val=audio[-int((val_len+test_len)/sf_EEG*sf_audio):-int(test_len/sf_EEG*sf_audio)] # (192000,)
            audio_test=audio[-int(test_len/sf_EEG*sf_audio):] # (192000,)

            EEG2=EEG_train
            audio2=audio_train
        else:
            EEG2 = EEG
            audio2 = audio
        #(80, 104301)

        # method 1: not good imshow plot
        '''
        audio2=np.int16(audio2 / np.max(np.abs(audio2)) * 32767) # (8772320,)
        windowLength=0.01
        mel2 = extractMelSpecs(audio2, sf_audio, windowLength=windowLength, frameshift=1/sf_EEG) # (548260, 23)
        mel2=mel2.transpose()
        start=int(windowLength/2*sf_EEG)
        EEG2=EEG2[:,start:start+mel2.shape[1]]
        '''
        # method 2: very good imshow plot
        mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024,
                                       n_mel_channels=mel_bins, sampling_rate=sf_audio, mel_fmin=0.0, mel_fmax=8000.0)
        audio2 = np.clip(audio2 / (np.max(np.abs(audio2))), -1, 1)  # (8772320,)
        # mel output lenght = math.ceil( (audio length)/hop_length ) (window was padded)
        # But, mel2 will have extra one point then the EEG2 data
        mel2 = mel_transformer.mel_spectrogram(torch.from_numpy(audio2)[np.newaxis, :].to(torch.float32)).squeeze().numpy()  # shape:([80, 104301]);

        ratio=EEG2.shape[1]/mel2.shape[1]
        steps=math.floor((EEG2.shape[1]-wind_size)/stride)
        for step in range(steps):
            stop_at=step*stride+wind_size
            x_session.append(EEG2[:,step*stride:stop_at]) # (118, 200)
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
