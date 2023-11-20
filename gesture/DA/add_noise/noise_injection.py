from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def noise_injection_3d(data,scale): # data in (trials, channels, time_point)
    data_NI=np.zeros((data.shape))
    for t in range(data_NI.shape[0]):
        data_NI[t,:,:]=noise_injection_(data[t,:,:],scale)
    return data_NI

# input data: (timepoint, channels)
def noise_injection_(data,scale):
    data=data.transpose()
    #trigger_original=data[:,-trigger_channel:]
    #data_original = data[: , :-trigger_channel]
    data_augmented = np.zeros(data.shape)
    fs = 1000

    # STFT
    for chi in range(data_augmented.shape[1]):
        nperseg = 300
        noverlap=150
        f, t, Zxx = signal.stft(data[:,chi], fs=fs, nperseg=nperseg,noverlap=noverlap)
        amp = np.abs(Zxx)
        angle = np.angle(Zxx)
        # amplitude-perturbation data augmentation
        mean=0
        gaussian = np.random.normal(loc=mean, scale=scale, size=(amp.shape[0], amp.shape[1]))
        amp_noisy=amp+gaussian
        Zxx_rec = amp_noisy * (np.cos(angle) + 1j * np.sin(angle))
        _, tmp = signal.istft(Zxx_rec, fs, nperseg=nperseg,noverlap=noverlap)
        data_augmented[:, chi]=tmp[:data_augmented.shape[0]] # there are few extra points generated, so truncate it;
        #shortest=tmp.shape[0] if tmp.shape[0]<data_augmented[:,chi].shape[0] else data_augmented[:,chi].shape[0]
        #plt.plot(data_original[:200,189])
        #plt.plot(tmp[:200])
        #plt.plot(data_augmented[:200,189])

    #trigger_data=np.concatenate((trigger_original,trigger_original),axis=0)
    #data=np.concatenate((data,trigger_data),axis=1)
    #data = np.concatenate((data_augmented, trigger_original), axis=1)

    return data_augmented.transpose()


