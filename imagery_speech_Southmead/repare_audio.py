from imagery_speech_Southmead.config import *
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt


filename = "C:/Users/xiaowu/mydrive/matlab/paradigms/imagery_speech_English_Southmead/audio/dSpeech_audio_encoding.wav"
print('Loading ' + filename + '.')
audio_sr, audio = wavfile.read(filename)

audio.max()

fig,ax=plt.subplots()
ax.plot(audio[:10000,:])
plt.show()
ax.clear()
sd.play(audio,audio_sr)
sd.stop()


