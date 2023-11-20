import numpy as np
import math
import matplotlib.pyplot as plt
from common_plot import *
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

from SOZ.config import *
from dsp.common_dsp import identify_noisy_channel

participant=participants[0]
soz_channels=soz_channel_all[participant] # ['B1-4', 'C1-5']

## load data
filename=data_dir+'raw_data/'+participant+'.edf'
raw_edf=mne.io.read_raw_edf(filename)
sf = int(raw_edf.info['sfreq'])

#### exclude channels by names
exclude_by_channel_name=exclude_channels[participant]
ch_names=[item for item in raw_edf.ch_names if item not in exclude_by_channel_name]
# raw_edf.get_channel_types()
raw_edf.pick_channels(ch_names)

####  exclude the noisy channel (50 Hz)
data=raw_edf.get_data()
noisy_channels=identify_noisy_channel(data, 50, 300, 1000)
noisy_ch_name=[raw_edf.ch_names[i[0]] for i in noisy_channels] # ['EEG F10-Ref']
ch_names=[item for item in raw_edf.ch_names if item not in noisy_ch_name]
raw_edf.pick_channels(ch_names)

#### filter out line noise
freqs = (50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950)
raw_notch = raw_edf.load_data().copy().notch_filter(freqs=freqs)
# visually inspect the notch result
fig1 = raw_edf.plot_psd(fmax=1000, average=True)
fig2 = raw_notch.plot_psd(fmax=1000, average=True)

####  visual check the data
mne.viz.plot_raw(raw_notch,n_channels=10) # pick (click) the bad channel
# raw_edf.info['bads'] # ['EEG F8-Ref', 'POL E']
raw_notch.drop_channels(raw_notch.info['bads'])

#### get the soz channel index
data=raw_notch.get_data()
soz_indexes=[] # [12, 13, 14, 15, 28, 29, 30, 31, 32]
for i in range(len(soz_channels)):
    soz_indexes.append([])
    soz_indexes[i]=np.where(np.asarray(raw_notch.info['ch_names']) == soz_channels[i])[0][0]
pick_one_soz=soz_indexes[0]
non_SOZ_index=np.setdiff1d(np.arange(len(raw_notch.ch_names)), soz_indexes).tolist()[:10] # memory full
all_indexes=non_SOZ_index+[pick_one_soz]
soz_index=all_indexes.index(pick_one_soz)#  np.where(all_indexes==pick_one_soz)

#### test SOZ (12) and non-SOZ(9,10,11) using 5s window
# total length: 368.123 s
start=3*60 # s
end=0
window=3 # s
step=1 # overlap for step s
trials=math.floor((data.shape[1]-start*sf-end*sf-window*sf)/(step*sf))

## try to use single trials
#data1=data[test_channels,int(start*sf):int((start+window)*sf)] # (4, 10000)
## try to use multiple trials
data2=np.zeros((len(all_indexes),trials,window*sf))  # (81, 185, 6000)
for i in range(trials):
    data2[:,i,:]=data[all_indexes,int((start+i*step)*sf):int((start+i*step+window)*sf)]


# (n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)
time_halfbandwidth_product=3
m = Multitaper(data2.swapaxes(0,2),
               #data1.T,
               sampling_frequency=sf,
               time_halfbandwidth_product=time_halfbandwidth_product,
               start_time=0)
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)

measures = dict(
    #pairwise_spectral_granger=c.pairwise_spectral_granger_prediction(), # (1, 500, 3, 3)
    #directed_transfer_function=c.directed_transfer_function(),  ##DTF
    #direct_directed_transfer_function=c.direct_directed_transfer_function(), ## dDTF
    #partial_directed_coherence=c.partial_directed_coherence(),  ## PDC
    generalized_partial_directed_coherence=c.generalized_partial_directed_coherence(), ## GPDC
    #coherence_magnitude=c.coherence_magnitude(),
)
DTF=measures['directed_transfer_function'].squeeze() # (1, 5000, 4, 4)
dDTF=measures['direct_directed_transfer_function'].squeeze()
PDC=measures['partial_directed_coherence'].squeeze() # (1, 5000, 4, 4)
gPDC=measures['generalized_partial_directed_coherence'].squeeze()
CohM=measures['coherence_magnitude'].squeeze() # (1, 5000, 4, 4)
measurement=gPDC
file_prefix='gPDC'
fig,ax=plt.subplots()
#### SOZ <-> non_SOZ: result not good at all
for i in range(len(non_SOZ_index)):
    ax.plot(c.frequencies, measurement[:, i, -1],linewidth=1, alpha=0.8,color=tab_colors_codes[0])
    ax.plot(c.frequencies, measurement[:, -1, i], linewidth=1, alpha=0.8,color=tab_colors_codes[1])
    ax.legend([str(pick_one_soz)+" -> "+str(non_SOZ_index[i]),str(non_SOZ_index[i])+" -> "+str(pick_one_soz)])
    filename=result_dir+file_prefix+'_'+str(pick_one_soz)+'_'+str(non_SOZ_index[i])
    fig.savefig(filename)
    ax.clear()




