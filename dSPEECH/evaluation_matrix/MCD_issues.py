from dSPEECH.config import *
import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
from scipy.io import wavfile
from dSPEECH.evaluation_matrix.utils.util import colored_line

folders = ['dataset1/reconstructed/mel_23/sid6/', 'dataset2/audio_samples/', 'dataset3/', 'dataset4/', 'dataset5/',
           'dataset6/', 'dataset7/', 'dataset8/', 'dataset9/', 'dataset10/']

target_files = []
for i in range(len(folders)):
    all_files_tmp = data_dir + 'evaluation_matrix/' + folders[i] + 'target_trial_*.wav'
    all_files = glob.glob(all_files_tmp)
    tmp = []

    for j in range(len(all_files)):
        file = os.path.normpath(all_files[j])
        tmp.append(file)
    target_files.append(tmp)

pred_files = []
for i in range(len(folders)):
    all_files_tmp = data_dir + 'evaluation_matrix/' + folders[i] + 'pred_trial_*.wav'
    all_files = glob.glob(all_files_tmp)
    tmp = []

    for j in range(len(all_files)):
        file = os.path.normpath(all_files[j])
        tmp.append(file)
    pred_files.append(tmp)

from mel_cepstral_distance import compare_audio_files

mcd_all = []
for i in range(len(target_files)):
    wave_files_target = target_files[i]  # per dataset
    wave_files_pred = pred_files[i]  # per dataset
    mcd_trials = []
    for j in range(len(wave_files_target)):
        trial_target = wave_files_target[j]  # per wave file
        trial_pred = wave_files_pred[j]

        mcd, penalty = compare_audio_files(trial_target, trial_pred)  # n_fft=512,win_len=512 (it's in ms, not samples)
        mcd_trials.append(mcd)
    mcd_all.append(mcd_trials)


