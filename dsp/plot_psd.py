import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import chirp

def plot_psd(data,sf=1000,title='Name me'):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / sf)[:N // 2]
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.title(title)
'''test plot_psd on chirp signal'''
fs=7200
Ts=1/fs
duration=4
t=np.arange(0,int(duration*fs))/fs
w=chirp(t,f0=1500,f1=250,t1=duration,method='quadratic')
plot_psd(w,sf=7200)

gen_sequence=gen_trial[0,:]
original_sequence=trial[0,:]
plot_psd(gen_sequence,title='Generated')
plot_psd(original_sequence,title='Original')