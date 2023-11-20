'''
This script generate all sort of signals
'''

from scipy.signal import chirp,spectrogram
import matplotlib.pyplot as plt
import numpy as np
#### generate chirp signals
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

fs=7200
T=4
t=np.arange(0,int(T*fs))/fs
w=chirp(t,f0=1500,f1=250,t1=T,method='quadratic') # f0是开始frequency， f1实在t1时刻的frequency。
ff,tt,Sxx=spectrogram(w,fs=fs,nperseg=256,nfft=576)
plt.pcolormesh(tt,ff[:145],Sxx[:145],cmap='gray_r',shading='gouraud')
plt.title("aTitle")
plt.xlabel('t (sec)')
plt.ylabel('Frequency (Hz)')
plt.grid()


