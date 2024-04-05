import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'Long': # 'Yoga'
    sys.path.extend(['C:/Users/xiaowu/mydrive/python/'])

from pre_all import computer, debugging, running_from_CMD
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import mne
from MIME_Huashan.config import data_dir
from pre_all import set_random_seeds
set_random_seeds(99999)
if running_from_CMD:
    sid = int(float(sys.argv[1]))
    scaler = sys.argv[2]
    time_stamp=sys.argv[3]
    testing=False
else:
    sid=1
    scaler='std'
    time_stamp='testing_time'
    testing=True
testing=(testing or debugging or computer=='mac')

sub_names = ['1_HHFU016_0714', '2_017_0725', '3_0815_Ma_JinLiang_0815', '4_HHFU22_0902', '5_HHFU026_1102',
             '6_HHFU027_motorimagery']
# 'HHFU027_motorimagery': a,b,c,d,e,f,g,h shafts; each have 8 contacts; 8*8=64 electrodes;
sub_name = sub_names[sid - 1]
sf=1000
filename = data_dir + 'preprocess/' + sub_name + '/raw_eeg_notched.fif'
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

if sid==1:
    # marker(marker_length=0.1)+rest_duration(3)+cue_duration(2) +fix_delay(1)+task_duraion(4)=10.1s
    epochs = mne.Epochs(raw, events, tmin=6.5, tmax=10.5, baseline=None)
elif sid==2 or sid==3 or sid==4 or sid==5 or sid==6:
    # rest_duration(3)+cue_duration(2)+marker(marker_length=0.1)+task_duraion(4)=10.1s
    epochs = mne.Epochs(raw, events, tmin=0, tmax=4, baseline=None)

#'MI':
# epoch1 = epochs['1'].get_data()
# epoch3 = epochs['3'].get_data()
# epoch5 = epochs['5'].get_data()
# epoch7 = epochs['7'].get_data()
# #'ME':
# epoch2 = epochs['2'].get_data()
# epoch4 = epochs['4'].get_data()
# epoch6 = epochs['6'].get_data()
# epoch8 = epochs['8'].get_data()
# #epoch_list_MI = [epoch1, epoch2, epoch3, epoch4]
# #epoch_list_ME = [epoch5, epoch6, epoch7, epoch8]
# total_len = epoch1.shape[2]

filename = data_dir + 'preprocess/' + sub_name + '/epoch1.fif'
epochs['1'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch2.fif'
epochs['2'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch3.fif'
epochs['3'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch4.fif'
epochs['4'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch5.fif'
epochs['5'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch6.fif'
epochs['6'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch7.fif'
epochs['7'].save(filename,overwrite=True)
filename = data_dir + 'preprocess/' + sub_name + '/epoch8.fif'
epochs['8'].save(filename,overwrite=True)
