import random

from MIME_Huashan.config import data_dir
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def read_data(sub_name='HHFU016',study='ME'):
    if study=='MI': # 1,3,5,7
        filename = data_dir + 'preprocess/' + sub_name + '/epoch1.fif'
        epoch1=mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch3.fif'
        epoch3 = mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch5.fif'
        epoch5 = mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch7.fif'
        epoch7 = mne.read_epochs(filename, preload=False)
        list_of_epochs = [epoch1.get_data(), epoch3.get_data(), epoch5.get_data(), epoch7.get_data()]
    elif study=='ME': # 2,4,6,8
        filename = data_dir + 'preprocess/' + sub_name + '/epoch2.fif'
        epoch2 = mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch4.fif'
        epoch4 = mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch6.fif'
        epoch6 = mne.read_epochs(filename, preload=False)
        filename = data_dir + 'preprocess/' + sub_name + '/epoch8.fif'
        epoch8 = mne.read_epochs(filename, preload=False)
        list_of_epochs = [epoch2.get_data(), epoch4.get_data(), epoch6.get_data(), epoch8.get_data()]

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

    return test_epochs, val_epochs, train_epochs