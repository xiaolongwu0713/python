import numpy as np
import matplotlib.pyplot as plt
from gesture.config import *
import hdf5storage
from gesture.utils import sub_sf_dict

info_dict=sub_sf_dict()
sids = [int(i) for i in list(info_dict.keys())]

for sid in sids:
    if sid != 4:
        continue
    print('Sid: '+str(sid)+'.')
    # sid=2
    sf=info_dict[str(sid)]
    if sf==2000: # already been downsampled in Matlab preprocessing1.
        sf=1000
    save_folder = data_dir + 'preprocessing_no_re_ref/' + 'P' + str(sid) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_folder = data_dir + 'preprocessing/' + 'P' + str(sid) + '/'
    data_path = data_folder + 'preprocessing1.mat'
    mat = hdf5storage.loadmat(data_path)
    datas = mat['Datacell']
    good_channels = mat['good_channels'] # sid4:60
    channelNum = len(np.squeeze(good_channels))

    raw_alls = []

    for session in range(2):

        # session=1
        data = datas[0, session]  # chn + emg1 +emg2 + emgdiff + trigger_label
        BCIdata = data[:, :-4]
        EMG = data[:, -4:-2]
        EMGdiff = data[:, -2]
        trigger_indexes = data[:, -1]

        ch_types = ['eeg', ] * BCIdata.shape[1]
        ch_names = ['eeg' + str(i) for i in range(BCIdata.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw_eeg = mne.io.RawArray(BCIdata.transpose(), info)

        ch_types = ['eeg']
        ch_names = ['emgdiff']
        info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw_emgdiff = mne.io.RawArray(EMGdiff[:, np.newaxis].transpose(), info)

        if sf == 1000:
            freqs = (50, 100, 150, 200, 250, 300, 350, 400, 450)
        else:
            freqs = (50, 100, 150, 200)
        raw_eeg.notch_filter(freqs=freqs)

        raw_emgdiff2 = raw_emgdiff.copy()
        emgdiff_env = raw_emgdiff.copy()
        raw_emgdiff2.filter(l_freq=20, h_freq=None)
        emgdiff_env.filter(l_freq=20, h_freq=None)

        emgdiff_env.apply_hilbert(envelope=True)

        from scipy.signal import savgol_filter

        ind = np.where(trigger_indexes > 0)
        env_smooth = savgol_filter(emgdiff_env.get_data()[0, :], 200, 3)  # window size 51, polynomial order 3
        env_smooth_deriv = []
        for i in range(0, len(env_smooth) - 1):
            # env_smooth_deriv.append(env_smooth[i]-2*env_smooth[i-1]+env_smooth[i-2])
            env_smooth_deriv.append(abs(env_smooth[i] - env_smooth[i - 1]))
        env_smooth_deriv.append(0)

        ind_emg = []
        if sid==12:
            threshold=15
        else:
            threshold=50
        for i in ind[0]:
            j = 0
            while True:
                if env_smooth_deriv[i + j] * 100 > threshold:  # env_smooth not reliable for some places. env_smooth_deriv is better.
                    ind_emg.append(i + j)
                    break
                j = j + 1
        if len(ind_emg)!=len(ind[0]):
            print('Triggers of EMG and EEG NOT the same.')
            sys.exit(1)
        # plot figure

        fig, ax = plt.subplots()
        ax.plot(raw_emgdiff2.get_data()[0, :],linewidth = '0.01')
        ax.plot(emgdiff_env.get_data()[0, :],linewidth = '0.01')
        ax.plot(env_smooth,linewidth = '0.01')
        ax.plot([i * 100 for i in env_smooth_deriv],linewidth = '0.01')
        for i in ind[0]:
            ax.axvline(x=i, color='green', linestyle='-',linewidth = '0.01')
        for i in ind_emg:
            ax.axvline(x=i, color='red', linestyle='--',linewidth = '0.01')
        ax.legend(['diff', 'diff_env', 'env_smooth', 'smooth_deriv(abs)'], fontsize=5)
        figname = save_folder + 'compare_trigger_session' + str(session) + '.eps'
        fig.savefig(figname,dpi=3000)

        sequence = [int(i) for i in trigger_indexes if i != 0]
        trigger_ind_emg = np.zeros(trigger_indexes.shape)
        for i, j in enumerate(ind_emg):
            trigger_ind_emg[j] = sequence[i]

        data_all = np.concatenate((raw_eeg.get_data(), EMG.transpose(), raw_emgdiff2.get_data(), emgdiff_env.get_data(),
                                   np.asarray(env_smooth_deriv)[np.newaxis, :], trigger_indexes[np.newaxis, :],
                                   trigger_ind_emg[np.newaxis, :]), axis=0)

        ch_types = ['eeg', ] * len(raw_eeg.ch_names) + ['misc', ] * 7
        ch_names = ['eeg' + str(i) for i in range(len(raw_eeg.ch_names))] + ['emg', 'emg', 'emg_diff', 'emg_diff_env',
                                                                             'env_smooth_deriv', 'trigger_index',
                                                                             'trigger_index_emg']
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        raw_all = mne.io.RawArray(data_all, info)

        raw_alls.append(raw_all)

    raw_final = mne.concatenate_raws(raw_alls)
    filename = save_folder + 'emg_trigger_raw.fif'
    raw_final.save(filename,overwrite=True)









