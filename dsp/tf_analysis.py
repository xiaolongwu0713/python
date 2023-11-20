'''
This script contain basic tf analysis examples
'''
from scipy.signal import chirp,spectrogram
import matplotlib.pyplot as plt
import numpy as np

#### FFT calculate PSD
#https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html

fs=7200
Ts=1/fs
duration=4
t=np.arange(0,int(duration*fs))/fs#ort=np.linspace(0.0,N*T,N,endpoint=False),Nisnumberofsamplepoints
N=len(t)
w=chirp(t,f0=1500,f1=250,t1=T,method='quadratic')
amp=fft(w) # 没有采样率和周期信息，fft函数不知道计算结果对应的频率
#with minus frequencies
freq=fftfreq(N,Ts) # （采样点数，采样周期）
plt.plot(freq,2.0/N*np.abs(amp))
#no minus frequencies
freq=fftfreq(N,Ts)[:N//2]
plt.plot(xf,2.0/N*np.abs(amp[0:N//2]))
