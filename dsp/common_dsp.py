import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal
import matplotlib.pyplot as plt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fbands2={}
fbands2['theta']=[1, 8]
fbands2['alpha']=[8, 12]
fbands2['beta1']=[12,20]
fbands2['beta2']=[20,32]
fbands2['gamma1']=[32,50]
fbands2['gamma2']=[50,100]
fbands2['gamma3']=[100,150]

'''
T=0.001#
fs=1/T
L=fs*10#10s
t=np.arange(0,L)*T

x=1*np.sin(2*np.pi*60*t)
x+=2*np.sin(2*np.pi*100*t)
x+=4*np.sin(2*np.pi*200*t)
x+=5*np.sin(2*np.pi*300*t)

x1=butter_bandpass_filter(x,0.1,100,fs,5)
t=np.arange(x.shape[0])
plt.plot(t,x1)

'''

# noisy_channel(data, 50, 300, 1000); data: [channel, timestamp]
def identify_noisy_channel(data, power_line_f, quality, fs, para=10):
    '''
    example call: noisy_channel(data, 50, 300, 1000); data: [channel, timestamp]
    '''
    b, a = signal.iirpeak(power_line_f, quality, fs=fs)
    output_signal = signal.filtfilt(b, a, data, axis=1)
    # noise_level = mean(sqrt(filter(b,a, signal).^2),1);
    noise_level = np.mean(np.sqrt(np.square(output_signal)), axis=1)
    plt.stem(noise_level)

    ## noisy channel
    mad=np.mean([abs(i - np.median(noise_level)) for i in noise_level])
    criterion=np.median(noise_level) + para * mad
    channel_tuple=np.where(noise_level>criterion)
    return channel_tuple[0]


