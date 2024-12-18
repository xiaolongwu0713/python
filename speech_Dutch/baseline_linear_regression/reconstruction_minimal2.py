## Used in dSPEECH.evaluation_matrix:
## This script is the same as reconstruction_minimal.py, except that this script output the test audio file.
## So that it is possible to compare the original test audio with the reconstructed test audio.

import soundfile
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pre_all import computer
import os
from pynwb import NWBHDF5IO
import numpy as np
from speech_Dutch.baseline_linear_regression.extract_features import extractHG,stackFeatures,extractMelSpecs
from speech_Dutch.baseline_linear_regression.reconstruction_minimal import createAudio
from scipy.stats import pearsonr

from speech_Dutch.config import data_dir

est = LinearRegression(n_jobs=5)
pca = PCA()

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
use_channels=False
winL=0.05
window_eeg=True
target_SR=16000
frameshift=0.01
modelOrder=4
stepSize=5
melbins=23
numComps = 50

sids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for sid in sids:
    folder = 'D:/data/BaiduSyncdisk/speech_Southmead/evaluation_matrix/dataset1/reconstructed/mel_' + str(melbins) + '/sid' + str(sid) + '/'
    print('Result folder: ' + folder + '.')
    # Load data
    participant = 'sub-' + f"{sid:02d}"
    filename = os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_ieeg.nwb')
    print('Loading ' + filename + '.')
    io = NWBHDF5IO(filename, 'r')
    nwbfile = io.read()
    # sEEG
    eeg = nwbfile.acquisition['iEEG'].data[:]  # (307523, 127)
    if use_channels:
        eeg = eeg[:, use_channels]
    eeg_sr = 1024
    # audio
    audio = nwbfile.acquisition['Audio'].data[:]  # (14414532,)
    audio_sr = 48000
    # target_SR = 16000
    # words (markers)
    words = nwbfile.acquisition['Stimulus'].data[:]
    words = np.array(words, dtype=str)  # (307511,)
    io.close()

    feat = extractHG(eeg, eeg_sr, windowLength=winL, frameshift=frameshift, window_eeg=window_eeg)

    import copy, resampy

    original_audio = copy.deepcopy(audio)
    audio = resampy.resample(audio, audio_sr, target_SR)  # 48000 to 22050

    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)  # 32767??

    # 4/5:1/5
    len1 = feat.shape[0]
    len2 = scaled.shape[0]
    breaker1 = int(len1 * 4 / 5)
    breaker2 = int(len2 * 4 / 5)
    feat_train, feat_test, scaled_train, scaled_test = feat[:breaker1, :], feat[breaker1:, :], scaled[:breaker2,
                                                                                               ], scaled[breaker2:, ]
    print('Stacking features using order: ' + str(modelOrder) + ',and step size: ' + str(stepSize) + '.')
    feat_train = stackFeatures(feat_train, modelOrder=modelOrder, stepSize=stepSize)  # (23980, 1143)
    feat_test = stackFeatures(feat_test, modelOrder=modelOrder, stepSize=stepSize)  # (5965, 1143)

    melSpec_train = extractMelSpecs(scaled_train, target_SR, melbins=melbins, windowLength=winL, frameshift=frameshift)
    melSpec_train = melSpec_train[modelOrder * stepSize:melSpec_train.shape[0] - modelOrder * stepSize, :]

    melSpec_test = extractMelSpecs(scaled_test, target_SR, melbins=melbins, windowLength=winL, frameshift=frameshift)
    melSpec_test = melSpec_test[modelOrder * stepSize:melSpec_test.shape[0] - modelOrder * stepSize, :]

    if melSpec_train.shape[0] != feat_train.shape[0]:
        print('Train: EEG and mel difference:' + str(melSpec_train.shape[0] - feat.shape[0]) + '.')
        tLen = np.min([melSpec_train.shape[0], feat_train.shape[0]])
        melSpec_train = melSpec_train[:tLen, :]  # (25863, 80)
        feat_train = feat_train[:tLen, :]  # (25863, 127)
    else:
        print('Train: EEG and mel lengths are the same. ')

    if melSpec_test.shape[0] != feat_test.shape[0]:
        print('Test: EEG and mel difference:' + str(melSpec_train.shape[0] - feat.shape[0]) + '.')
        tLen = np.min([melSpec_test.shape[0], feat_test.shape[0]])
        melSpec_test = melSpec_test[:tLen, :]  # (25863, 80)
        feat_test = feat_test[:tLen, :]  # (25863, 127)
    else:
        print('Test: EEG and mel lengths are the same. ')

    # feat_train, feat_test, melSpec_train, melSpec_test, scaled_test

    mu = np.mean(feat_train, axis=0)
    std = np.std(feat_train, axis=0)
    trainData = (feat_train - mu) / std
    testData = (feat_test - mu) / std

    pca.fit(trainData)
    trainData = np.dot(trainData, pca.components_[:numComps, :].T)
    testData = np.dot(testData, pca.components_[:numComps, :].T)

    est.fit(trainData, melSpec_train)
    rec_spec = est.predict(testData) # (5961, 23)

    r,p=pearsonr(rec_spec,melSpec_test,axis=0)
    r_avg=r.mean()
    print('Average CC: '+str(r_avg)+'.')
    audiosr = 16000
    winLength = winL
    frameshift = frameshift
    pred2 = createAudio(rec_spec, audiosr=audiosr, winLength=winLength, frameshift=frameshift)
    truth2 = scaled_test # createAudio(truth, audiosr=audiosr, winLength=winLength, frameshift=frameshift)
    #np.save(folder + 'waveform_truth.npy', truth2)
    #np.save(folder + 'waveform_pred.npy', pred2)

    soundfile.write(folder + 'waveform_truth.wav', truth2, audiosr)
    soundfile.write(folder + 'waveform_pred.wav', pred2, audiosr)