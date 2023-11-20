import random

from MIME_Huashan.config import data_dir
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def read_data(sub_name='HHFU016',study='ME',scaler='std'):
    sf=1000
    filename = data_dir + 'preprocess/' + sub_name + '/raw_eeg_processed.fif'
    print('Load '+filename+'.')
    raw = mne.io.read_raw_fif(filename,preload=True)
    filename = data_dir + 'preprocess/' + sub_name + '/events.eve'
    events = mne.read_events(filename)

    data=raw.get_data()
    data=data.transpose()
    if scaler=='std':
        print("Standard scaler.")
        scaler = StandardScaler()
        data = scaler.fit_transform((data))
    elif scaler=='minmax':
        print("Minmax scaler.")
        scaler = MinMaxScaler(feature_range=(0,1))
        dataa = scaler.fit_transform((data))
    else:
        print('No scaler.')

    chn_names = ["seeg"] * data.shape[1]
    chn_types = ["seeg"] * data.shape[1]
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=sf)
    raw = mne.io.RawArray(data.transpose(), info)

    # marker+marker_length(0.1)+rest_duration(3)+cue_duration(2)+fix_delay(1) # +task_duraion(4)=10.1s
    # 6.1-10.1
    if sub_name=='HHFU016':
        epochs = mne.Epochs(raw, events, tmin=6.5, tmax=10.5, baseline=None)
    else:
        epochs = mne.Epochs(raw, events, tmin=0, tmax=4, baseline=None)

    if study=='MI':
        epoch1 = epochs['1'].get_data()
        epoch2 = epochs['3'].get_data()
        epoch3 = epochs['5'].get_data()
        epoch4 = epochs['7'].get_data()
    elif study=='ME':
        epoch1 = epochs['2'].get_data()
        epoch2 = epochs['4'].get_data()
        epoch3 = epochs['6'].get_data()
        epoch4 = epochs['8'].get_data()
    list_of_epochs = [epoch1, epoch2, epoch3, epoch4]
    total_len = list_of_epochs[0].shape[2]

    # validate=test=2 trials
    class_number=4
    trial_number = [list(range(epochi.shape[0])) for epochi in list_of_epochs]  # [ [0,1,2,...19],[0,1,2...19],... ]
    test_trials = [random.sample(epochi, 2) for epochi in trial_number]  # randomly choose two trial as the test dataset
    # len(test_trials[0]) # test trials number
    trial_number_left = [np.setdiff1d(trial_number[i], test_trials[i]) for i in range(class_number)]

    val_trials = [random.sample(list(epochi), 2) for epochi in trial_number_left]
    train_trials = [np.setdiff1d(trial_number_left[i], val_trials[i]).tolist() for i in range(class_number)]

    # no missing trials
    assert [sorted(test_trials[i] + val_trials[i] + train_trials[i]) for i in range(class_number)] == trial_number

    test_epochs = [epochi[test_trials[clas], :, :] for clas, epochi in
                   enumerate(list_of_epochs)]  # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
    val_epochs = [epochi[val_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]
    train_epochs = [epochi[train_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]

    return test_epochs, val_epochs, train_epochs, scaler