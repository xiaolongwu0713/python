'''
This is the script creating ECoG triggers. There are two type triggers: one has three consecutive audio pulses; another
has only one audio pulse;
'''
from librosa import load
from pre_all import mydrive
import matplotlib.pyplot as plt
import soundfile
## consecutive squarewave
#file=mydrive+'D:\data\speech_Southmead\audio\square_wave\15_second_wavs_ONE_SQUARE_WAVE\1.wav'
file=mydrive+'matlab/paradigms/speech_Southmead/audio/square_wave/15_second_wavs_ONE_SQUARE_WAVE/'+'1.wav'
#audio, sf = load(file)
audio, sf = load(file,sr=48000)
fig,ax=plt.subplots()
ax.plot(audio)
one_p=audio[:222]

sf=48000 # 22050/48000
low_duration=1
three_squares=[0.01]*int(sf*low_duration)+list(one_p)+[0.01]*int(sf*low_duration)+list(one_p)+[0.01]*int(sf*low_duration)+list(one_p)+[0.01]*int(sf*low_duration)
ax.clear()
ax.plot(three_squares)
filename=mydrive+'matlab/paradigms/speech_Southmead/audio/square_wave/three_squarewaves.wav'
soundfile.write(filename, three_squares, sf)

one_square=[0]*int(sf*1)+list(one_p)+[0]*int(sf*1)
filename=mydrive+'matlab/paradigms/speech_Southmead/audio/square_wave/one_squarewave.wav'
soundfile.write(filename, one_square, sf)

