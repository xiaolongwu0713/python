import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy
from scipy.signal import chirp
from scipy.fft import fft,fftfreq


w = pywt.Wavelet('bior6.8')
(dec_lo, dec_hi, rec_lo, rec_hi)=w.filter_bank

plt.plot(dec_lo)
plt.plot(dec_hi)
plt.plot(rec_lo)
plt.plot(rec_hi)

# copied from http://wavelets.pybytes.com/wavelet/bior6.8/   Coefficients/Decomposition low-pass filter
dec_lo2=[0.0,0.0019088317364812906,-0.0019142861290887667,-0.016990639867602342,0.01193456527972926,0.04973290349094079,-0.07726317316720414,
-0.09405920349573646,0.4207962846098268,0.8259229974584023,0.4207962846098268,-0.09405920349573646,-0.07726317316720414,
0.04973290349094079,0.01193456527972926,-0.016990639867602342,-0.0019142861290887667,0.0019088317364812906]
# they are equal
assert(dec_lo==dec_lo2)

(phi_d, psi_d, phi_r, psi_r, x) = w.wavefun(level=5)


pywt.wavelist('db') # list all available wavelet matching db*
w = pywt.Wavelet('db10')
(dec_lo, dec_hi, rec_lo, rec_hi)=w.filter_bank
(phi_d, psi_d, x) = w.wavefun(level=5)

from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np

# plot the frequency response
# indeed it is a low pass filter
N=len(dec_lo)
yf = fft(dec_lo)
xf = fftfreq(N, 1/1000)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

# indeed it is a high pass filter
N=len(dec_hi)
yf = fft(dec_hi)
xf = fftfreq(N, 1/1000)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))


# test the filter bank on chirp signal
fs=1000
Ts=1/fs
T=4
t=np.arange(0,int(T*fs))/fs # or t=np.linspace(0.0, N*T, N, endpoint=False), N is number of sample points
N=len(t)
original=chirp(t,f0=1,f1=500,t1=T,method='quadratic')
amp=fft(original)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp[0:N//2]),color='r')

# conv with dec_lo
filt1l=np.convolve(original,dec_lo,mode='same')
amp1l=fft(filt1l)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp1l[0:N//2]),color='g')

filt1h=np.convolve(original,dec_hi,mode='same')
amp1h=fft(filt1h)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp1h[0:N//2]),color='g')


# no down sample. Low pass the same previous low pass range if no down sampling. Exactly the same as previous effect.
filt2l=np.convolve(filt1l,dec_lo,mode='same')
amp2l=fft(filt2l)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp2l[0:N//2]),color='b')

filt2h=np.convolve(filt1l,dec_hi,mode='same')
amp2h=fft(filt2h)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp2h[0:N//2]),color='b')

# down sample. Low pass the lower half of previous low pass range.
filt1l=filt1l[::2,]
fs=500
Ts=1/fs
filt3l=np.convolve(filt1l,dec_lo,mode='same')
amp3l=fft(filt3l)
N3=len(filt3l)
freq = fftfreq(N3, Ts)[:N3//2]
plt.plot(freq, 2.0/N3 * np.abs(amp3l[0:N3//2]),color='y')

filt3h=np.convolve(filt1l,dec_hi,mode='same')
amp3h=fft(filt3h)
N3=len(filt3h)
freq = fftfreq(N3, Ts)[:N3//2]
plt.plot(freq, 2.0/N3 * np.abs(amp3h[0:N3//2]),color='y')


# up sample. FFT doesn't change after the upsampling.
# So to keep sampe number constent: original(N points)-- low pass--down sample(2/N)--up sample the low-passed signal(N)--repeat.
fs=1000
Ts=1/fs
filt3l_up=scipy.signal.resample(filt3l,2*len(filt3l))
filt3h_up=scipy.signal.resample(filt3h,2*len(filt3h))
N=len(filt3l_up)
amp3l_up=fft(filt3l_up)
freq = fftfreq(N, Ts)[:N//2]
plt.plot(freq, 2.0/N * np.abs(amp3l_up[0:N//2]),color='aqua')




