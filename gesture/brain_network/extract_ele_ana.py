## This script extract the anatomic name/label and position (electrodes_Final_Norm.mat obtained by freesurfer) of each electrode into a dict, ele.
## The new name is in the format of: "sid-lab_id-total/i-th": for example, 2-3020-4/1: the 1st electrode (out of 4) of sid2 in the 3020 region;
## The correspondance between SEEG data and electrode anatomic info is achieved by SignalChanel_Electrode_Registration.mat, which was
## further explained below:
import numpy as np

# the anatomic index (in file 'electrodes_Final_Norm.mat') is the row index of the channel number of the good_channels variable
# for example, for a channel 35 from good_channels, with the following reg_file as below, this channel corresponds to the 4th
# row of the electrodes_Final_Norm.mat file;
# 1 22
# 2 33
# 3 1
# 4 35
# 5 36

from gesture.utils import get_epoch
import scipy.io
from gesture.config import *
import h5py
import pandas as pd
from gesture.brain_network.util import reduce_duplicated_hyph

# parse the look_up_table into ana_dic (dict): key(name)-->value(index)
ana_dic={}
filename=meta_dir+'EleCTX_Files/FsTutorial_AnatomicalROI_FreeSurferColorLUT.txt'
with open(filename,"r") as infile:
    lines=infile.read().split('\n')
for line in lines:
    if len(line)>0: # not an empty line
        if line[0].isdigit():
            tmp=line.split()
            ana_dic[tmp[1]]=tmp[0] #
ana_dic['ctx-nown']='9999' # patient anatomic file contain 'ctx-nown' label which is not contained by the look_up_table

info_file=info_dir+'Info(use_Info.txt_instead).npy' # TODO: use the Info.txt instead
info=np.load(info_file,allow_pickle=True)

filename=meta_dir+"good_channels.mat"
mat=scipy.io.loadmat(filename)
good_channels_tmp=mat['good_channels'][0]
good_channels={}
reg={}
ele={}

for i,j in enumerate(info[:,0]):
    #sid=good_channels_tmp[i].dtype.names[0]
    sid='sid'+str(j)
    ele[sid] = {}

    reg_file = meta_dir + 'EleCTX_Files/P' + str(j) + '/SignalChanel_Electrode_Registration.mat'
    tmp = scipy.io.loadmat(reg_file)
    reg[sid] = list(tmp['CHN'][:, 0]) # channel index in a list

    tmp=good_channels_tmp[i][0][sid][0][0,:]
    tmp_list=[int(m) for m in tmp]
    good_channels[sid]=tmp_list # sid2: 115 channels

    good_channels_indices=[reg[sid].index(l) for l in good_channels[sid]] # good channel index in the registration file

    filename=meta_dir+'EleCTX_Files/P'+str(j)+'/electrodes_Final_Norm.mat'
    f=h5py.File(filename, 'r')

    channel_number = f['elec_Info_Final_wm']['name'].shape[0] # sid2: 121 in total (bad and good)
    for key in ['ana_label_name', 'name']:
        value_list = [] # 121
        for k in range(channel_number):
            #a=np.asarray(f[np.asarray(f['elec_Info_Final_wm'][key])[k, 0]])
            a=f[f['elec_Info_Final_wm'][key][k,0]]
            value=''.join(chr(i[0]) for i in a)
            # ctx--lh-superiorparietal in ana_name_list2 is not contained in ana_dic
            value=reduce_duplicated_hyph(value)
            value_list.append(value)
        ana_name_list2=[value_list[n] for n in good_channels_indices] # extract 115 channels from total 121
        ele[sid][key] = ana_name_list2
        if key=='ana_label_name':
            # !!!! make sure to keep the original channel sequence
            # Also can be converted to anatomical index using the freesurfer look_up_table
            ana_id_list=[ana_dic[keyy] for keyy in ana_name_list2]
            # parse the label into "sid-lab_id-total/i-th": for example, 2-3020-4/1: the 1st electrode (out of 4) of sid2 in 3020 region;
            counting={x: ana_id_list.count(x) for x in set(ana_id_list)}
            occurrence={x:0 for x in set(ana_id_list)}
            ana_id_list2=[]
            for lab_id in ana_id_list:
                occurrence[lab_id]+=1
                tmp=str(j)+'_'+lab_id+'_'+str(counting[lab_id])+'/'+str(occurrence[lab_id])
                ana_id_list2.append(tmp)
            ele[sid]['ana_label_id']=ana_id_list2

    for key in ['ano_pos', 'norm_pos', 'pos']:
        tmp_list = []
        for k in range(channel_number):
            #a=np.asarray(f[np.asarray(f['elec_Info_Final_wm'][key])[k, 0]])
            a = f[f['elec_Info_Final_wm'][key][k, 0]]
            #tmp_list.append(''.join(chr(i[0]) for i in a))
            value=[i.item() for i in a]# [104.0, 134.0, 138.0]
            tmp_list.append(value)
        tmp_list2 = [tmp_list[n] for n in good_channels_indices]
        ele[sid][key]=tmp_list2

filename=meta_dir+'ele_anat_position_info.npy'
np.save(filename, ele)
    # sid41 doesn't have below info
    # for key in ['GMPI', 'LocalPTD', 'PTD', 'ana_cube']:
    #     tmp2=np.array(f['elec_Info_Final_wm'][key])
    #     ele[sid][key] = tmp2

#Total information: (['GMPI', 'LocalPTD', 'PTD', 'ana_cube', 'ana_label_name', 'ano_pos', 'name', 'norm_pos', 'pos'])
#filename=meta_dir+'ele_anat_position_info.npy'
#ele2 = np.load(filename,allow_pickle='TRUE').item()

sid=2
fs=1000
selected_channels=False
# 4 extra channels: [2*emg + 1*trigger_indexes + 1*emg_trigger]
#test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid, fs, selected_channels=selected_channels,scaler='std',EMG=True)
epochs=get_epoch(sid, fs,scaler='std',EMG=True,tmin=-4,tmax=5)
epoch1=epochs['0']# 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1']
epoch3=epochs['2']
epoch4=epochs['3']
epoch5=epochs['4']

epoch1.plot(n_epochs=5, n_channels=5, scalings=dict(seeg=3))

# populate new channel name
info_tmp=epoch1.info
old=info_tmp.ch_names
new=ele['sid2']['ana_label_id']
mapping={old[i]:new[i] for i in range(len(new))}
mne.rename_channels(info_tmp,mapping)
epoch1.info=info_tmp

epoch1.plot_psd(picks=['seeg'])
epoch=epoch1.copy().load_data().filter(l_freq=0.01,h_freq=3,picks=['seeg'])

matrixes=[]
for tmin in np.arange(-4,5,1):
    data=epoch.get_data(picks=['seeg'],tmin=tmin,tmax=tmin+1)
    for trial in data:
        dataframe=pd.DataFrame(data=trial.transpose(),columns=ele['sid2']['ana_label_id'])
        matrix = dataframe.corr()
        matrixes.append(matrix)
matrix_avg=np.average(np.asarray(matrixes),axis=0)

plt.imshow(matrix_avg, cmap='Blues')
plt.colorbar()
variables=ele['sid2']['ana_label_id']
plt.xticks(range(len(matrix)), variables, rotation=90, ha='right')
plt.yticks(range(len(matrix)), variables)
