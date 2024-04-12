import mne
from MIME_Huashan.config import data_dir
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import numpy as np

fig,ax=plt.subplots()

filename=data_dir+'preprocess/6_HHFU027_motorimagery/raw_emg.fif'
raw=mne.io.read_raw_fif(filename,preload=True)
raw.compute_psd().plot()
raw.notch_filter(np.arange(50, 451, 50), picks=['all'])
filt_raw = raw.copy().filter(l_freq=60, h_freq=100,picks='all')
filt_raw.plot()
raw.plot()
regexp = r"(POL B3[1234])"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),)

info_bak=raw.info
ch_types=['eeg']*len(raw.ch_names)
info = mne.create_info(ch_names=raw.ch_names, ch_types=ch_types, sfreq=raw.info['sfreq'])
raw = mne.io.RawArray(raw.get_data(), info)
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None,picks='all')

ica = ICA(n_components=8, max_iter="auto", random_state=97)
ica.fit(filt_raw,picks='all')

explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )

ica.plot_sources(raw, show_scrollbars=False)
ica.exclude = [0,] # visually check the first component is the ECG signal
reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.compute_psd().plot()

#emg=emg.get_data() # (8, 2177037)
ax.plot(emg[-1,10*1000:20*1000])
ax.clear()


