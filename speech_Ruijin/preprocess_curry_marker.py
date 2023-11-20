'''
use the first part of this script to visually check the data (bad channel etc.)
'''

from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

from speech_Ruijin.config import *
import matplotlib.pyplot as plt
import mne
import os
from scipy.io import wavfile
import textgrid
from speech_Ruijin.config import extra_EEG_extracting,extra_EEG_pairring
from example_speech.closed_loop_seeg_speech_synthesis_master_bak.local.offline import compute_spectrogram
from example_speech.closed_loop_seeg_speech_synthesis_master_bak.local.offline import herff2016_b
import librosa

sids=[1,2,3]
sessions=2 # two sessions
sf_T_EEG=1/sf_EEG
sf_T_audio=1/sf_audio

# extra_EEG_extracting: add extra_EEG_extracting EEG data before and after audio
# attach the audio data to the EEG data for sanity check. however the audio waveform changed so much after downsample that it is impossible to make the comparison;
def extract_EEG(sid,attach_audio=False):
#for sid in sids:
    ele_count=None
    EEG_range=[]
    audio_length_marker=None
    skip_corrupt_EEG=0
    if sid==1:
        EEG_start=["2023/4/17 14:53:07:997","2023/4/17 15:06:28:002"]
        mark10_begin=["2023/4/17 14:54:07:878","2023/4/17 15:07:27:330"]
        ele_count=128 # ele_count is the real SEEG channels # Total: 195=ele_count+surface EEG and 1 stim;
        detrend=[1,46,47,]
        #bad_channels = [1,17,28, 29, 30,31,32,46,47,48, 69,70,71, 72, 80, 87, 88, 95, 96, 111,125,127]
        bad_channels = [17,28,29,48,71,72,87,88,95,96,]
        # first few seconds data is corrupted for both EEG recordings
        skip_corrupt_EEG = 3 #2.5  # s
    elif sid==2:
        EEG_start = ["2023/4/30 15:08:29:824", "2023/4/30 15:08:29:824"]
        mark10_begin = ["2023/4/30 15:08:52:644", "2023/4/30 15:20:39:749"]
        mark10_end = ["2023/4/30 15:16:31:482", "2023/4/30 15:27:53:885"]
        ele_count=72 # Total: 84
        bad_channels = [17,18,31,64,65,73,74,75,76,77,78,79,80]
        # first few seconds data is corrupted for both EEG recordings
        skip_corrupt_EEG = 2.5  # s
    elif sid==3:
        EEG_start = ["2023/6/21 14:05:27:352", "2023/6/21 14:05:27:352"]
        mark10_begin = ["2023/6/21 14:15:49:436", "2023/6/21 14:30:19:813"]
        mark10_end = ["2023/6/21 14:22:55:592", "2023/6/21 14:37:26:521"]
        ele_count=84 # Total: 84
        bad_channels = [39,40,77,80,81,82,83,84]
        # first few seconds data is corrupted for both EEG recordings
        skip_corrupt_EEG = 5  # s
    elif sid==4:
        EEG_start = ["2023/7/16 13:59:12:946", "2023/7/16 13:59:12:946"]
        mark10_begin = ["2023/7/16 14:00:39:516", "2023/7/16 14:10:17:044"]
        mark10_end = ["2023/7/16 14:07:29:621", "2023/7/16 14:15:48:542"]
        ele_count=84 # Total: 84
        bad_channels = [39,40,77,80,81,82,83,84]
        # first few seconds data is corrupted for both EEG recordings
        skip_corrupt_EEG = 5  # s

    channels_all=[*range(1,ele_count+1)]
    for session in range(sessions):
        tmp1 = EEG_start[session] # EEG recording start
        tmp2=mark10_begin[session] # EEG clip beginning
        format = "%Y/%m/%d %H:%M:%S:%f"
        tmp1=datetime.strptime(tmp1, format)
        tmp2=datetime.strptime(tmp2, format)
        offset=(tmp2-tmp1).total_seconds() * 1000
        if sid != 1:
            tmp1 = mark10_begin[session]  # EEG recording start
            tmp2 = mark10_end[session]  # EEG clip beginning
            format = "%Y/%m/%d %H:%M:%S:%f"
            tmp1 = datetime.strptime(tmp1, format)
            tmp2 = datetime.strptime(tmp2, format)
            audio_length_marker = (tmp2 - tmp1).total_seconds() * 1000

        #### read EEG data
        if sid==1:
            file = data_dir + "P" + str(sid) + "/raw/EEG/curry/session" + str(session + 1) + "/export_to_edf.edf"
        else:
            file = data_dir + "P" + str(sid) + "/raw/EEG/curry/export_to_edf.edf"
        raw = mne.io.read_raw_edf(file) # include=ele_name
        raw.pick(picks=[str(c) for c in channels_all])
        raw.load_data().notch_filter(np.arange(50, 251, 50))
        # inspect the data and exclude bad channels
        raw.filter(l_freq=1,h_freq=None)
        if 1==0:
            raw.plot()

        #offset=int(audio_offset-EEG_offset-skip_corrupt_EEG*sf_EEG) # 56885 #use this value to align EEG data with the audio data

        good_channels=[c for c in channels_all if c not in bad_channels]
        channel_name=[str(i) for i in good_channels]
        data=raw.get_data(picks=channel_name) # shape: (118 channel, 653000 time)

        #### read audio data
        filename = data_dir+ "P" + str(sid)+"/processed/audio/session" + str(session+1)+ "_denoised.wav"
        #filename2 = os.path.join(data_dir, "P" + str(sid), "raw", "audio","session" + str(session + 1), "recording.wav")
        sf_audio, audio = wavfile.read(filename)


        audio_length = audio.shape[0] / sf_audio  # 556.27 s
        '''
        attached_audio=librosa.resample(np.float32(audio),orig_sr=sf_audio,target_sr=sf_EEG)
        attached_audio=[np.ones((offset,))*attached_audio.mean(),attached_audio,np.ones((data.shape[1]-offset-attached_audio.shape[0]))*attached_audio.mean()]
        attached_audio=np.concatenate(attached_audio,axis=0)
        data_aug=np.concatenate((data,attached_audio[np.newaxis,:]),axis=0)
        if 1==0:
            plt.plot(audio, label="Left channel")
            fig, ax=plt.subplots()
        if attach_audio:
            data=data_aug
        else:
            data=data
        '''

        #check if audio_length is equal to the marker (10) difference
        if sid==1: # P1 don't have audio stop marker.
            print("No audio length: no audio ending marker: extract EEG data according to the audio length.")
        else:
            try:
                audio_error=audio_length*sf_EEG-audio_length_marker # -838
                print("sid"+str(sid)+"/session"+str(session)+": audio_error: "+str(audio_error)+" ms.")
                assert audio_error==0 # 32 ms
            except AssertionError: # P2 marked the autdio end too early.
                print("Audio time error: "+str(audio_error)+"ms.")
                ''' should mark '10' after the PsychPortAudio('Close')???
                original implementation: 
                % Stop recording:
                PsychPortAudio('Stop', pahandle);
                % EEG stop marker
                io64(ioObj,address,10);
                io64(ioObj,address,0);
                % Perform a last fetch operation to get all remaining data from the capture engine:
                audiodata = PsychPortAudio('GetAudioData', pahandle);
                % Attach it to our full sound vector:
                recordedaudio = [recordedaudio audiodata];
                % Close the audio device:
                PsychPortAudio('Close', pahandle);
                '''
            # is there a better way to align EEG and audio?

        data=data[:,int(offset-extra_EEG_extracting*sf_EEG):int(offset+audio_length*sf_EEG+extra_EEG_extracting*sf_EEG)]

        folder = data_dir + "P" + str(sid) + "/processed/EEG/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder + "session" + str(session + 1) + ".npy"
        np.save(filename, data)
        print('Saving to: '+filename+'.')

        '''
        ch_types = ['eeg'] * data.shape[0]
        ch_names = ['eeg'] * data.shape[0]
        info = mne.create_info(ch_names=ch_names, sfreq=sf_EEG, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        #raw = raw.filter(70, 170)  # (118, 558270)
        filename = folder + "session" + str(session + 1) + ".npy"
        print('Saving to '+filename+'.')
        np.save(filename, raw.get_data())
        '''

        '''
        ## spectrogram looks not right
        ## test the feature extraction and selection
        print("Feature extraction and selection.")
        data=data[:,extra_EEG_extracting*sf_EEG:-extra_EEG_extracting*sf_EEG]
        x_train = herff2016_b(data.transpose(), sf_EEG, 0.05, 0.01)  # (118, 490500)
        y_train = compute_spectrogram(audio, sf_audio, 0.016, 0.01)
        y_train = y_train[20:-4] # (40, 48826)
        select = feature_selection(x_train, y_train)  # 150
        x_train = x_train[:, select].transpose() # (590, 48826)

        mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80,
                                       sampling_rate=48000, mel_fmin=0.0, mel_fmax=8000.0)


        folder = data_dir + "P" + str(sid) + "/processed/dataset/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder + "session" + str(session + 1) + "_EEG_features.npy"
        np.save(filename, x_train)
        filename = folder + "session" + str(session + 1) + "_mel_features.npy"
        np.save(filename, y_train.transpose())
        '''

        # no need to save the audio
        #folder = data_dir+"P" + str(sid)+ "/processed/audio/"
        #if not os.path.exists(folder):
        #    os.makedirs(folder)
        #filename = folder+ "session" + str(session + 1) + ".npy"
        #np.save(filename, audio)

#### Note that the begining of the EEG data is corrupted, around 500 ms;
#filename=r'D:\new\Long\edf\ddd1.edf'
#raw=mne.io.read_raw_edf(filename)

# for each pair, there are extra_EEG_pairring EEG before and after audio onset and offset
def pairring(sid,extra_EEG_pairring=extra_EEG_pairring):
    for session in range(sessions):
        # EEG data
        filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session + 1) + ".npy"
        EEG = np.load(filename)
        filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session + 1) + "_highgamma.npy"
        highgamma = np.load(filename)
        # audio data
        filename = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session + 1) + "_denoised.wav"
        sf_audio, audio = wavfile.read(filename)
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

        # convert grid timestamp to index
        pairs = []
        for inter in intervals:
            pair = []
            # audio indexing
            index1 = int(inter[0] / sf_T_audio)
            index2 = int(inter[1] / sf_T_audio)
            audio_clip = audio[index1:index2, ]
            # audio_clip2 =audio_clip/np.max(np.abs(audio_clip))
            pair.append(audio_clip)
            # EEG indexing
            index1 = int((inter[0] + extra_EEG_extracting-extra_EEG_pairring) / sf_T_EEG)
            index2 = int((inter[1] + extra_EEG_extracting+extra_EEG_pairring) / sf_T_EEG)
            EEG_clip = EEG[:, index1:index2]  # (118, 1080)
            highgamma_clip=highgamma[:, index1:index2]
            # EEG_clip2 = scaler.fit_transform(EEG_clip.transpose())
            # pair.append(EEG_clip2.transpose())
            pair.append(EEG_clip)
            pair.append(highgamma_clip)
            pairs.append(pair)

        # bad trials, such as burst wave, low-frequency drift, reading error
        pairs_bad = []
        for inter in intervals_bad:
            pair = []
            # audio indexing
            index1 = int(inter[0] / sf_T_audio)
            index2 = int(inter[1] / sf_T_audio)
            audio_clip = audio[index1:index2, ]
            # audio_clip2 = audio_clip / np.max(np.abs(audio_clip))
            pair.append(audio_clip)
            # EEG indexing
            index1 = int((inter[0] + extra_EEG_extracting) / sf_T_EEG)
            index2 = int((inter[1] + extra_EEG_extracting) / sf_T_EEG)
            EEG_clip = EEG[:, index1:index2]
            # EEG_clip2 = scaler.fit_transform(EEG_clip.transpose())
            # pair.append(EEG_clip2.transpose())
            pair.append(EEG_clip)
            pairs_bad.append(pair)

        filename = data_dir + "P" + str(sid) + "/processed/dataset/session" + str(session + 1) + "_pairs.npy"
        np.save(filename, np.array(pairs, dtype=object), allow_pickle=True)
        filename = data_dir + "P" + str(sid) + "/processed/dataset/session" + str(session + 1) + "_pairs_bad.npy"
        np.save(filename, np.array(pairs_bad, dtype=object), allow_pickle=True)
        # load list
        # b = np.load(filename, allow_pickle=True)
        # pairi=b[0,:]


if __name__=="__main__":
    sids=[1,2,3,4]
    for sid in sids:
        #sid=4
        extract_EEG(sid)
        #pairring(sid)







