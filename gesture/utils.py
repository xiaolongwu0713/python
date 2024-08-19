import glob

import hdf5storage
import random

import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from gesture.DA.add_noise.noise_injection import  noise_injection_3d
from gesture.config import *

# this function returns dict with 'sid2' as key and channel index list [1,2,3,4....] as value;
def get_good_channels():
    filename=meta_dir+"good_channels.mat"
    mat=scipy.io.loadmat(filename)
    good_channels_tmp=mat['good_channels'][0]
    good_channels={}

    for i in range(len(good_channels_tmp)):
        key=good_channels_tmp[i][0].dtype.names[0]
        value=good_channels_tmp[i][0].item()[0][0]
        good_channels[key]=value
    return good_channels

# This function returns dict with key as 'sid2' and value as 1000;
def sub_sf_dict():
    info_file=info_dir+'Info.txt' # TODO: use the Info.txt instead
    info=[]
    with open(info_file,'r') as f:
        lines=f.read().split('\n')
        for line in lines:
            tmp=[]
            if len(line)>0:
                line=line.partition(',')
                tmp.append(int(line[0]))
                tmp.append(int(line[2]))
                info.append(tmp)

    info_dict={}
    for i in info:
        info_dict[str(i[0])]=i[1]

    return info_dict

## read good sids
def read_good_sids():
    outfile=meta_dir+'good_sids.txt'
    with open(outfile,"r") as infile:
        lines=infile.read().split('\n')
    good_sids=[int(tmpi) for tmpi in lines if len(tmpi)>0] # 确保读到的有数据，而不是空string
    return good_sids

def read_sids():
    outfile=meta_dir+'info/info.txt'
    with open(outfile,"r") as infile:
        lines=infile.read().split('\n')
    sids_sf=[tmpi for tmpi in lines if len(tmpi)>0] # 确保读到的有数据，而不是空string
    sids=[int(tmpi.partition(",")[0]) for tmpi in sids_sf]
    return sids

def read_channel_number():
    channel_number={}
    outfile = meta_dir + 'info/channel_number.txt'
    with open(outfile, "r") as infile:
        lines_tmp = infile.read().split('\n')
    lines = [tmpi for tmpi in lines_tmp if len(tmpi) > 0]  # 确保读到的有数据，而不是空string
    for sid_record in lines:
        sid = sid_record.partition(",")[0]
        numb= sid_record.partition(",")[2]
        channel_number[sid]=int(numb)
    return channel_number

# method: original/NI/WGAN_GP/WGAN/VAE
def get_decoding_acc(sids, method,wind=500,stride=200):
    acc_dict = {}
    modeli = 'deepnet'
    for sid in sids:
        if method in ['NI', 'WGAN_GP', 'WGAN', 'DCGAN', 'VAE']:
            result_folder = result_dir + 'deepLearning/retrain/DA_' + method + '/' + str(sid) + '/'
            result_file = result_folder + modeli + '_'+ str(wind)+'_'+str(stride) + '.npy'
        elif method == 'vanilla':
            result_folder = result_dir + 'deepLearning/vanilla/' + str(wind)+'_'+str(stride)+'/'+str(sid) + '/'
            result_file = result_folder + modeli + '_'+ str(wind)+'_'+str(stride) + '.npy'
        elif method=='FBCSP':
            result_file = result_dir + 'machineLearning/FBCSP/' + str(sid) + '.npy'
        result = np.load(result_file, allow_pickle=True).item()
        if method !='FBCSP': # FBCSP does not have training_loss or val_acc or train_acc
            train_losses = result['train_losses']
            train_accs = result['train_accs']
            val_accs = result['val_accs']
        acc_dict[str(sid)] = result['test_acc']

    return acc_dict

def read_data(sid, fs, selected_channels=None,scaler='std',augment_with_noise_std=None):
    # read data
    data_folder= data_dir+'preprocessing/'+'P'+str(sid)+'/'
    data_path = data_folder+'preprocessing2.mat'
    mat=hdf5storage.loadmat(data_path)
    data = mat['Datacell']
    good_channels=mat['good_channels']
    channelNum=len(np.squeeze(good_channels))
    #channelNum=int(mat['channelNum'][0,0])
    # total channel = channelNum + 4[2*emg + 1*trigger_indexes + 1*emg_trigger]
    data=np.concatenate((data[0,0],data[0,1]),0)
    del mat

    if augment_with_noise_std=='before':
        data=noise_injection(data,4)
    # standardization
    # StandardScaler will not clamp data between -1 and 1, but MinMaxScaler will [0,1].
    if scaler=='std':
        print("Standard scaler.")
        chn_data=data[:,-4:]
        dataa=data[:,:-4] # (1052092, 208)
        scaler = StandardScaler()
        dataa = scaler.fit_transform((dataa))
        data=np.concatenate((dataa,chn_data),axis=1)
    elif scaler=='minmax':
        print("Minmax scaler.")
        chn_data = data[:, -4:]
        dataa = data[:, :-4]  # (1052092, 208)
        scaler = MinMaxScaler(feature_range=(0,1))
        dataa = scaler.fit_transform((dataa))
        data = np.concatenate((dataa, chn_data), axis=1)
    else:
        print('No scaler.')
    if augment_with_noise_std=='after':
        data=noise_injection(data,4)
    # stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
    chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
    chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
    raw = mne.io.RawArray(data.transpose(), info)

    # gesture/events type: 1,2,3,4,5; session: 1 1 2 33 4 2 1 445 3 422 1 1 23 3 4: 5 3D cube: (time,channel, trial)
    events0 = mne.find_events(raw, stim_channel='stim_trigger')
    events1 = mne.find_events(raw, stim_channel='stim_emg')
    # events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
    events0=events0-[0,0,1]
    events1=events1-[0,0,1]

    #print(events[:5])  # show the first 5
    # Epoch from 4s before(idle) until 4s after(movement) stim1.
    raw=raw.pick(["seeg"])

    if selected_channels:
        raw = raw.pick(selected_channels)
        print('MNE picking selected channels.')

    epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)
    # or epoch from 0s to 4s which only contain movement data.
    # epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

    epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
    epoch2=epochs['1'].get_data()
    epoch3=epochs['2'].get_data()
    epoch4=epochs['3'].get_data()
    epoch5=epochs['4'].get_data()
    list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
    total_len=list_of_epochs[0].shape[2]

    # validate=test=2 trials
    trial_number=[list(range(epochi.shape[0])) for epochi in list_of_epochs] #[ [0,1,2,...19],[0,1,2...19],... ]
    test_trials=[random.sample(epochi, 2) for epochi in trial_number] # randomly choose two trial as the test dataset
    # len(test_trials[0]) # test trials number
    trial_number_left=[np.setdiff1d(trial_number[i],test_trials[i]) for i in range(class_number)]

    val_trials=[random.sample(list(epochi), 2) for epochi in trial_number_left]
    train_trials=[np.setdiff1d(trial_number_left[i],val_trials[i]).tolist() for i in range(class_number)]

    # no missing trials
    assert [sorted(test_trials[i]+val_trials[i]+train_trials[i]) for i in range(class_number)] == trial_number

    test_epochs=[epochi[test_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)] # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
    val_epochs=[epochi[val_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]
    train_epochs=[epochi[train_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]

    return test_epochs, val_epochs, train_epochs,scaler

def noise_injection_epoch_list(train_epochs,std_scale):
    train_epochs_NI=list()
    for cli in range(len(train_epochs)):
        tmp=noise_injection_3d(train_epochs[cli],std_scale)
        train_epochs_NI.append(tmp)
    return train_epochs_NI

# epochs could be created using EEG trigger channel (trigger='EEG') or it could be created by checking the EMG signals (trigger='EMG')
# Nots abou the EMG created epochs
# 1. It is not re-referenced, so it's suitable for correlation analysis of different brain regions.
# 2. It contains 'bad channels': data were not extracted by good_channels as in the preprocess2.m file.
def get_epoch(sid, fs, selected_channels=None,scaler='std',cv_idx=None,EMG=False,tmin=0,tmax=5,trigger='EEG',random_shift=False):
    if trigger.upper() == 'EEG':
        # read data
        data, channelNum = read_data_(sid)  # data: (1052092, 212)
        if scaler!='no':
            data, scalerr = norm_data(data, scaler=scaler)
        epochs = gen_epoch(data, fs, channelNum, selected_channels=selected_channels, EMG=EMG,tmin=tmin,tmax=tmax)
        return epochs
    elif trigger.upper()=='EMG':
        save_folder = data_dir + 'preprocessing_no_re_ref/' + 'P' + str(sid) + '/'
        filename = save_folder + 'emg_trigger_raw.fif'
        raw=mne.io.read_raw_fif(filename)
        events_eeg = mne.find_events(raw, stim_channel='trigger_index')
        events_emg = mne.find_events(raw, stim_channel='trigger_index_emg')
        if random_shift is not False:
            shift = [int(random.uniform(random_shift[0], random_shift[1])) for _ in range(events_eeg.shape[0])]
            events_eeg[:,0] = events_eeg[:,0]+shift
            events_emg[:,0] = events_emg[:, 0] + shift

        if len(events_emg)==len(events_eeg):
            epochs_eeg = mne.Epochs(raw, events_eeg, tmin=tmin, tmax=tmax, baseline=None)
            epochs_emg = mne.Epochs(raw, events_emg, tmin=tmin, tmax=tmax, baseline=None)
        else:
            sys.exit("Events number obtained by EEG and EMG are not the same!!!")
        return epochs_emg

def read_data_split_function(sid, fs, selected_channels=None,scaler='std',cv_idx=None,EMG=False,trigger='EEG'):
    data,channelNum=read_data_(sid) # data: (1052092, 212)
    if scaler != 'no':
        data, scalerr = norm_data(data, scaler=scaler)
    epochs=gen_epoch(data, fs, channelNum, selected_channels=selected_channels, EMG=EMG)
    test_epochs, val_epochs, train_epochs = data_split(epochs, cv_idx=cv_idx)
    return test_epochs, val_epochs, train_epochs, scalerr


# rest(4 s)--> cue (1s) ---> movement (5 s)
# tmin, tmax=0,5;
def gen_epoch(data,fs,channelNum,selected_channels=None, EMG=False,tmin=0,tmax=5):
    chn_names=np.append(["eeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
    chn_types=np.append(["eeg"]*channelNum,["emg","emg","stim","stim"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
    raw = mne.io.RawArray(data.transpose(), info)

    # gesture/events type: 1,2,3,4,5; session: 1 1 2 33 4 2 1 445 3 422 1 1 23 3 4: 5 3D cube: (time,channel, trial)
    events0 = mne.find_events(raw, stim_channel='stim_trigger')
    events1 = mne.find_events(raw, stim_channel='stim_emg')
    # events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
    events0=events0-[0,0,1]
    events1=events1-[0,0,1]

    #print(events[:5])  # show the first 5
    # Epoch from 4s before(idle) until 4s after(movement) stim1.
    if EMG:
        picks=["eeg","emg"]
    else:
        picks=["eeg"]
    raw=raw.pick(picks) #raw=raw.pick(["seeg"])

    if selected_channels:
        raw = raw.pick(selected_channels)
        print('MNE picking '+str(len(selected_channels))+' selected channels.')
    # stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
    # or epoch from 0s to 4s which only contain movement data.
    # epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)
    epochs = mne.Epochs(raw, events1, tmin=tmin, tmax=tmax, baseline=None)
    return epochs

def read_data_(sid):
    data_folder = data_dir + 'preprocessing/' + 'P' + str(sid) + '/'
    data_path = data_folder + 'preprocessing2.mat'
    mat = hdf5storage.loadmat(data_path)
    data = mat['Datacell']
    good_channels = mat['good_channels']
    channelNum = len(np.squeeze(good_channels))
    # channelNum=int(mat['channelNum'][0,0])
    # total channel = channelNum + 4[2*emg + 1*trigger_indexes + 1*emg_trigger]
    data = np.concatenate((data[0, 0], data[0, 1]), 0)
    del mat
    return data,channelNum

# standardization
# StandardScaler will not clamp data between -1 and 1, but MinMaxScaler will [0,1].
# 4[2*emg + 1*trigger_indexes + 1*emg_trigger]
def norm_data(data,scaler='std'):
    if scaler=='std':
        print("Standard scaler.")
        chn_data=data[:,-4:]
        dataa=data[:,:-4] # (1052092, 208)
        scaler = StandardScaler()
        dataa = scaler.fit_transform((dataa))
        data=np.concatenate((dataa,chn_data),axis=1)
    elif scaler=='minmax':
        print("Minmax scaler.")
        chn_data = data[:, -4:]
        dataa = data[:, :-4]  # (1052092, 208)
        scaler = MinMaxScaler(feature_range=(0,1))
        dataa = scaler.fit_transform((dataa))
        data = np.concatenate((dataa, chn_data), axis=1)
    else:
        print('No scaler.')
    return data,scaler

def data_split(epochs,cv_idx=None):
    epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
    epoch2=epochs['1'].get_data()
    epoch3=epochs['2'].get_data()
    epoch4=epochs['3'].get_data()
    epoch5=epochs['4'].get_data()
    list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
    total_len=list_of_epochs[0].shape[2]

    # validate=test=2 trials
    trial_number=[list(range(epochi.shape[0])) for epochi in list_of_epochs] #[ [0,1,2,...19],[0,1,2...19],... ]

    if not cv_idx==None:
        print("Using CV method.")
        test_idx=[[0,1],[4,5],[8,9],[12,13],[16,17]]
        val_idx = [[2,3], [6,7], [10,11], [14,15], [18,19]]
        test_trials=[test_idx[cv_idx]] * 5
        trial_number_left = [np.setdiff1d(trial_number[i], test_trials[i]) for i in range(class_number)]
        val_trials = [val_idx[cv_idx]] * 5
        train_trials = [np.setdiff1d(trial_number_left[i], val_trials[i]).tolist() for i in range(class_number)]
    else:
        test_trials=[[7, 4], [19, 11], [13, 3], [19, 2], [13, 11]]#test_trials=[random.sample(epochi, 2) for epochi in trial_number] # randomly choose two trial as the test dataset
        # len(test_trials[0]) # test trials number
        trial_number_left=[np.setdiff1d(trial_number[i],test_trials[i]) for i in range(class_number)]

        val_trials=[[2, 13], [17, 2], [14, 6], [5, 15], [18, 14]] #val_trials=[random.sample(list(epochi), 2) for epochi in trial_number_left]
        train_trials=[np.setdiff1d(trial_number_left[i],val_trials[i]).tolist() for i in range(class_number)]

    # no missing trials
    assert [sorted(test_trials[i]+val_trials[i]+train_trials[i]) for i in range(class_number)] == trial_number

    test_epochs=[epochi[test_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)] # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
    val_epochs=[epochi[val_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]
    train_epochs=[epochi[train_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]

    return test_epochs, val_epochs, train_epochs

def read_gen_data(sid,method,time_stamp,scaler='std',cv=1):
    print("Reading generated data.")
    #data_folder = data_dir + 'preprocessing/' + 'P' + str(sid) + '/'+method+'/'
    data_folder = 'D:/tmp/python/gesture/DA/'+method+'/sid' + str(sid) + '/cv' + str(cv) + '/' + time_stamp+ '/Samples/'
    gen_class=[]
    if scaler=='std':
        scaler = StandardScaler()
    for clas in range(5):
        tmp_file = data_folder + 'class' + str(clas)+ '_cv'+str(cv)+'.npy'
        #tmp_file = os.path.normpath(glob.glob(tmp_file)[0])
        tmp = np.load(tmp_file, allow_pickle=True) # (760, 5, 500)
        tmp2=np.zeros((tmp.shape))
        for i, trial in enumerate(tmp):
            scaler = StandardScaler()
            dataa = scaler.fit_transform((trial.transpose()))
            tmp2[i]=dataa.transpose()
        gen_class.append(tmp2)
    print('Reading augmented data done.')
    return gen_class

def windowed_data(train_epochs, val_epochs, test_epochs, wind, stride, gen_data_all=None, train_mode=None, method=None):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for clas, epochi in enumerate(test_epochs):
        Xi, y = slide_epochs(epochi, clas, wind, stride)
        assert Xi.shape[0] == len(y)
        X_test.append(Xi)
        y_test.append(y)
    X_test = np.concatenate(X_test, axis=0)  # (1300, 63, 500)
    y_test = np.asarray(y_test)
    y_test = np.reshape(y_test, (-1, 1))  # (5, 270)

    for clas, epochi in enumerate(val_epochs):
        Xi, y = slide_epochs(epochi, clas, wind, stride)
        assert Xi.shape[0] == len(y)
        X_val.append(Xi)
        y_val.append(y)
    X_val = np.concatenate(X_val, axis=0)  # (1300, 63, 500)
    y_val = np.asarray(y_val)
    y_val = np.reshape(y_val, (-1, 1))  # (5, 270)

    for clas, epochi in enumerate(train_epochs):
        Xi, y = slide_epochs(epochi, clas, wind, stride)  # (576, 208, 500)
        assert Xi.shape[0] == len(y)
        if train_mode == 'DA' and method != 'NI':
            Xi = np.concatenate((Xi, gen_data_all[clas]), axis=0)
            y = [clas] * Xi.shape[0]
            if clas == 0:
                print("Adding the generated data to the training dataset !! ")
        X_train.append(Xi)
        y_train.append(y)
    X_train = np.concatenate(X_train, axis=0)  # (1300, 63, 500)
    y_train=np.concatenate(y_train)
    y_train=y_train[:,np.newaxis]
    #y_train = np.asarray(y_train)  # sum(y_train==4)
    #y_train = np.reshape(y_train, (-1, 1))  # (5, 270)-->(-1, ?)
    return X_train, y_train, X_val, y_val, X_test, y_test