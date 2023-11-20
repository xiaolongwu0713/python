'''
align EEG and audio using the trigger line exported from curry
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


sids=[5,]
sessions=2 # two sessions
sf_T_EEG=1/sf_EEG
sf_T_audio=1/sf_audio
sid=5
session=0

#### read EEG data
file = data_dir+ "P" + str(sid)+"/raw/EEG/session"+ str(session+1) + "_curry.edf"
raw = mne.io.read_raw_edf(file) # include=ele_name
raw.load_data().notch_filter(np.arange(50, 251, 50))
raw.filter(l_freq=1,h_freq=None) # optional
# inspect the data and exclude bad channels
bad_channels=raw.info['bads']
raw.drop_channels(bad_channels) # pick and drop the bad channels

events=mne.events_from_annotations(raw)

'''
(array([[ 60036,      0,      2],
        [ 64900,      0,      4],
        [ 64996,      0,      3],
        [ 67620,      0,      3],
        [ 70244,      0,      3],
        [ 72900,      0,      3],
        [ 75524,      0,      3],
        [ 78148,      0,      2],
        [137458,      0,      2],
        [144531,      0,      4],
        [144691,      0,      3],
        [147348,      0,      3],
        [149972,      0,      3],
        [152620,      0,      3],
        [155245,      0,      3],
        [157902,      0,      3],
        [160528,      0,      3],
        [163121,      0,      3],
        [165778,      0,      3],
        [168404,      0,      3],
        [171029,      0,      3],
        [173654,      0,      3],
        [176312,      0,      3],
        [178937,      0,      3],
        [181562,      0,      3],
        [184220,      0,      3],
        [186909,      0,      3],
        [189535,      0,      3],
        [192192,      0,      3],
        [194849,      0,      3],
        [197443,      0,      3],
        [200100,      0,      3],
        [202757,      0,      3],
        [205383,      0,      3],
        [208040,      0,      3],
        [210633,      0,      3],
        [213256,      0,      3],
        [215942,      0,      3],
        [218565,      0,      3],
        [221219,      0,      3],
        [223874,      0,      3],
        [226528,      0,      3],
        [229151,      0,      3],
        [231805,      0,      3],
        [234460,      0,      3],
        [237114,      0,      3],
        [239737,      0,      3],
        [242391,      0,      3],
        [245078,      0,      3],
        [247732,      0,      3],
        [250387,      0,      3],
        [253041,      0,      3],
        [255696,      0,      3],
        [258350,      0,      3],
        [261037,      0,      3],
        [263691,      0,      3],
        [266346,      0,      3],
        [268969,      0,      3],
        [271691,      0,      3],
        [274347,      0,      3],
        [277067,      0,      3],
        [279723,      0,      3],
        [282379,      0,      3],
        [285035,      0,      3],
        [287692,      0,      3],
        [290380,      0,      3],
        [293036,      0,      3],
        [295692,      0,      3],
        [298348,      0,      3],
        [301004,      0,      3],
        [303692,      0,      3],
        [306316,      0,      3],
        [309004,      0,      3],
        [311660,      0,      3],
        [314380,      0,      3],
        [317037,      0,      3],
        [319693,      0,      3],
        [322349,      0,      3],
        [325005,      0,      3],
        [327661,      0,      3],
        [330381,      0,      3],
        [333014,      0,      3],
        [335669,      0,      3],
        [338356,      0,      3],
        [341011,      0,      3],
        [343666,      0,      3],
        [346321,      0,      3],
        [348976,      0,      3],
        [351663,      0,      3],
        [354287,      0,      3],
        [356974,      0,      3],
        [359693,      0,      3],
        [362348,      0,      3],
        [364971,      0,      3],
        [367690,      0,      3],
        [370345,      0,      3],
        [373000,      0,      3],
        [375719,      0,      3],
        [378374,      0,      3],
        [381029,      0,      3],
        [383716,      0,      3],
        [386372,      0,      3],
        [389027,      0,      3],
        [391703,      0,      3],
        [394327,      0,      3],
        [396983,      0,      3],
        [399608,      0,      3],
        [402264,      0,      3],
        [404920,      0,      3],
        [407576,      0,      3],
        [410200,      0,      3],
        [412888,      0,      3],
        [415512,      0,      3],
        [418168,      0,      3],
        [420824,      0,      3],
        [423480,      0,      3],
        [426136,      0,      2],
        [461379,      0,      1],
        [461379,      0,      1]]),
 {'1': 1, '10': 2, '20': 3, '8': 4})
 
 marker '10' is 2 now.
 '''
start=137458
end=426136
eeg=raw.get_data()[:,int(start-extra_EEG_extracting*sf_EEG):int(end+extra_EEG_extracting*sf_EEG)] # (109, 288678)
file = data_dir+ "P" + str(sid)+"/raw/EEG/session"+ str(session+1) + "_curry_crop.npy"
np.save(file,eeg)




