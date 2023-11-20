import os
from pathlib import Path
import natsort
import numpy as np
from sEMG_Zhangbin.config import *
channel_number=7
hands_number=2
task_number=15
fs=2000
def data_cutter2(i, data, emg_channel):
    data=data[2*fs:,emg_channel] # discard first 2 seconds
    if 4<=i<=9: # task 5-10 last for 10s
        if data.shape[0] > 10 * fs: # pick 10s between 2-12 s
            print("Cut: " + str((data.shape[0] - 10 * fs) / fs) + ".")
            data = data[:10 * fs-1,]
        else:
            print('Less than 12s:'+str(data.shape[0]/fs+2)+'s.')
    else: # other tasks last for 20s
        if data.shape[0] > 20 * fs: # pick 10s between 2-22 s
            print("Cut: " + str((data.shape[0] - 20 * fs) / fs) + ".")
            data = data[:20 * fs-1,]
        else:
            print('Less than 22s:'+str(data.shape[0]/fs+2)+'s.')
    return data

def data_cutter1(i,data, emg_channel):
    fs=2000
    if 4 <= i <= 7:
        if data.shape[0] > 12 * fs: # pick 10s between 2-12 s
            print("Cut: " + str((data.shape[0] - 10 * fs) / fs) + ".")
            data = data[2*fs:12 * fs-1, emg_channel]
        else:
            print("less than 12 s.")
            data = data[2*fs:, emg_channel]
    elif 8<=i<=9: # pick 10s between (1,11)
        if data.shape[0] > 11 * fs:
            print("Cut: " + str((data.shape[0] - 10 * fs) / fs) + ".")
            data = data[1*fs:11* fs - 1, emg_channel]
        else:
            print("less than 11 s.")
            data = data[1*fs:, emg_channel]
    elif 10<=i<=12: # pick 5s between (1,6)
        if data.shape[0] > 6 * fs:
            print("Cut: " + str((data.shape[0] - 5 * fs) / fs) + ".")
            data = data[1*fs:6* fs - 1, emg_channel]
        else:
            print("less than 6 s.")
            data = data[1*fs:, emg_channel]
    else: # pick 20s between (2,22)
        if data.shape[0] > 22 * fs:
            print("Cut: " + str((data.shape[0] - 20* fs) / fs) + ".")
            data = data[2*fs:22* fs - 1, emg_channel]
        else:
            print("less than 22 s.")
            data = data[2*fs:, emg_channel]
    return data


def get_ET_data(taski=None):
    raw_ET={}
    for p in ET:
        print('Reading ' + p + '.')
        raw_ET[p] = {}
        for handi in ['L','R']:
            handi_trials=[]
            sub_folder=data_dir+'Data-'+handi+'/'+p+'/'
            if os.path.exists(sub_folder):
                trial_files = natsort.realsorted([str(pth) for pth in Path(sub_folder).iterdir() if pth.suffix == '.txt'])
                tmp=trial_files
                #assert len(trial_files)==15
                trial_numbers=[Path(i).name[6:8] for i in trial_files]
                if taski is not None:
                    if len(trial_files)==15:
                        tmp=[trial_files[i] for i in taski]
                    else:
                        break
                for i,trial_file in enumerate(tmp):
                    data=np.genfromtxt(trial_file, delimiter='\t')[:-1,:-1] # because of the incomplete last row and an extra column filled with nan value
                    # EMG channels
                    if data.shape[1]==22: # 1+7+7+7
                        if handi=='L':
                            emg_channel=range(1,8)
                        elif handi=='R':
                            emg_channel=range(8,15)
                    elif data.shape[1]==15: # 1+7+7
                        emg_channel=range(1,8)
                    else:
                        raise NameError('Channel number is not 22 or 15.')

                    data = data_cutter(i,data, emg_channel)
                    handi_trials.append(data)
                raw_ET[p][handi]=handi_trials

    return raw_ET

def get_PD_data(taski=None):
    raw_PD={}
    for p in PD:
        print('Reading ' + p + '.')
        raw_PD[p] = {}
        for handi in ['L','R']:
            handi_trials=[]
            sub_folder=data_dir+'Data-'+handi+'/'+p+'/'
            if os.path.exists(sub_folder):
                trial_files = natsort.realsorted([str(pth) for pth in Path(sub_folder).iterdir() if pth.suffix == '.txt'])
                tmp = trial_files
                trial_numbers=[Path(i).name[6:8] for i in trial_files]
                if taski is not None:
                    if len(trial_files) == 15:
                        tmp = [trial_files[i] for i in taski]
                    else:
                        break
                for i,trial_file in enumerate(tmp):
                    data=np.genfromtxt(trial_file, delimiter='\t')[:-1,:-1] # because of the incomplete last row and an extra column filled with nan value
                    # EMG channels
                    if data.shape[1] == 22:  # 1+7+7+7
                        if handi == 'L':
                            emg_channel = range(1, 8)
                        elif handi == 'R':
                            emg_channel = range(8, 15)
                    elif data.shape[1] == 15:  # 1+7+7
                        emg_channel = range(1, 8)

                    data=data_cutter(i,data, emg_channel)
                    handi_trials.append(data)
                raw_PD[p][handi]=handi_trials

    return raw_PD

def get_others_data(taski=None):
    raw_others={}
    for p in others:
        print('Reading ' + p + '.')
        raw_others[p] = {}
        for handi in ['L','R']:
            handi_trials=[]
            sub_folder=data_dir+'Data-'+handi+'/'+p+'/'
            if os.path.exists(sub_folder):
                trial_files = natsort.realsorted([str(pth) for pth in Path(sub_folder).iterdir() if pth.suffix == '.txt'])
                tmp = trial_files
                trial_numbers=[Path(i).name[6:8] for i in trial_files]
                if taski is not None:
                    if len(trial_files) == 15:
                        tmp = [trial_files[i] for i in taski]
                    else:
                        break
                for i,trial_file in enumerate(tmp):
                    data=np.genfromtxt(trial_file, delimiter='\t')[:-1,:-1] # because of the incomplete last row and an extra column filled with nan value
                    # EMG channels
                    if data.shape[1] == 22:  # 1+7+7+7
                        if handi == 'L':
                            emg_channel = range(1, 8)
                        elif handi == 'R':
                            emg_channel = range(8, 15)
                    elif data.shape[1] == 15:  # 1+7+7
                        emg_channel = range(1, 8)

                    data = data_cutter(i,data, emg_channel)
                    if p=='TP006' and handi=='R':
                        data[:,5]=data[:,6] # channe 5th is not connected
                    handi_trials.append(data)
                raw_others[p][handi]=handi_trials

    return raw_others


# NC filename: TN010;
def get_NC_data(taski=None):
    raw_NC={}
    for p in NC:
        print('Reading '+p+'.')
        raw_NC[p] = {}
        for handi in ['L','R']:
            handi_trials=[]
            sub_folder=data_dir+'Data-'+handi+'/'+p+'/'
            if os.path.exists(sub_folder):
                trial_files = natsort.realsorted([str(pth) for pth in Path(sub_folder).iterdir() if pth.suffix == '.txt'])
                tmp = trial_files
                trial_numbers=[Path(i).name[-6:-4] for i in trial_files]
                if taski is not None:
                    if len(trial_files) == 15:
                        tmp = [trial_files[i] for i in taski]
                    else:
                        break
                for i,trial_file in enumerate(tmp):
                    data=np.genfromtxt(trial_file, delimiter='\t')[:-1,:-1] # because of the incomplete last row and an extra column filled with nan value
                    # EMG channels
                    if data.shape[1] == 22:  # 1+7+7+7
                        if handi == 'L':
                            emg_channel = range(1, 8)
                        elif handi == 'R':
                            emg_channel = range(8, 15)
                    elif data.shape[1] == 15:  # 1+7+7
                        emg_channel = range(1, 8)

                    data = data_cutter(i,data, emg_channel)
                    handi_trials.append(data)
                raw_NC[p][handi]=handi_trials
    return raw_NC

def get_data(class_name,taski=None):
    if class_name=='ET':
        data=get_ET_data(taski=taski)
    elif class_name=='PD':
        data=get_PD_data(taski=taski)
    elif class_name=='NC':
        data=get_NC_data(taski=taski)
    elif class_name=='others':
        data=get_others_data(taski=taski)
    return data

data_cutter=data_cutter2
# one particular task data
extract_one_task=False
taski = [0]  # a list of task indexes
if extract_one_task:
    strr = ""
    for i in taski:
        strr = strr + str(i) + "_"
    strr = strr[:-1]
    ET_data=get_data('ET',taski=taski)
    PD_data=get_data('PD',taski=taski)
    others_data=get_data('others',taski=taski)
    NC_data=get_data('NC',taski=taski)

    print("Save to "+data_dir+".")
    np.save(data_dir+'ET_data_task_'+strr+'.npy', ET_data)
    np.save(data_dir+'PD_data_task_'+strr+'.npy', PD_data)
    np.save(data_dir+'others_data_task_'+strr+'.npy', others_data)
    np.save(data_dir+'NC_data_task_'+strr+'.npy', NC_data)

else:
    # all tasks data
    ET_data=get_data('ET')
    PD_data=get_data('PD')
    others_data=get_data('others')
    NC_data=get_data('NC')

    #data_dir = 'H:/Long/data/sEMG_Zhangbin/data2/'
    #data_dir='/Volumes/Samsung_T5/data/sEMG_zhangbin/data2/'
    data_dir=data_dir+'data2/'
    np.save(data_dir+'ET_data.npy', ET_data)
    np.save(data_dir+'PD_data.npy', PD_data)
    np.save(data_dir+'others_data.npy', others_data)
    np.save(data_dir+'NC_data.npy', NC_data)







