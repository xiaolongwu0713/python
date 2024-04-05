'''
this is a simplified paradigm using pseudoword, instead of real words sentences.
3 phases: Listen-->speak-->image.
Each phase begins with a 0.01s square wave and a 0.2s C6 beep.
Sentences are not equal in length.
Patients are expected to overtly/covertly speak immediately after the beep.
'''

import numpy as np
import librosa
from dSPEECH.config import *
from librosa import load
from librosa.util import fix_length
import os
import copy
import soundfile
from pathlib import Path
from natsort import realsorted
import resampy

#import librosa.display # this package messed up the plt package;
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
# We'll need IPython.display's Audio widget
from IPython.display import Audio


import sounddevice as sd
sf=48000
sd.play(audioss,sf)
sd.stop()
ax.plot(audioss)


folder=mydrive+'matlab/paradigms/speech_Southmead/Scott/Bath-dSPEECH/dSPEECH_audio/pseudowords/3_vowels_audio/'
file_list=realsorted([str(pth) for pth in Path(folder).iterdir() if pth.suffix=='.wav'])
sf=48000
for file in file_list:
    result=folder+Path(file).name.partition("-")[0]+'.wav'
    audios=[]
    # listen
    # 0.9 second before the actual audio
    audios.append(np.zeros(int(sf * .3)))
    audios.append(np.ones(int(sf * .01))) # 0.01 second squre wave
    audios.append(librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.2) / 4)  # beep
    # actual audio
    audio, asf = load(file, sr=48000, mono=False)
    if asf != 48000:
        audio=resampy.resample(audio, asf, 48000) # 48000 to 22050
    length = audio.shape[0]
    audios.append(audio)

    # speak
    audios.append(np.ones(int(sf * .01)))  # 0.01 second squre wave
    audios.append(librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.2) / 4)  # beep
    audios.append(np.zeros(length))

    # image
    audios.append(np.ones(int(sf * .01)))  # 0.01 second squre wave
    audios.append(librosa.tone(librosa.note_to_hz('C6'), sr=sf, duration=0.2) / 4)  # beep
    audios.append(np.zeros(length))

    audioss = np.concatenate(audios)
    soundfile.write(result, audioss, sf)


aa=mydrive+'matlab/paradigms/speech_Southmead/audio/square_wave/15_second_wavs_ONE_SQUARE_WAVE/one_squarewave.wav'
b, asf = load(aa, sr=48000,mono=False)

