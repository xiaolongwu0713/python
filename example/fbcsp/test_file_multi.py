
#%%
from sklearn.preprocessing import StandardScaler

from example.fbcsp.FBCSP_support_function import loadDatasetD2, computeTrialD2, createTrialsDictD2, loadTrueLabel, loadDatasetD2_Merge
from example.fbcsp.FBCSP_Multiclass import FBCSP_Multiclass
import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC,SVC
from sklearn.calibration import CalibratedClassifierCV
from scipy.io import loadmat
import time
import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
from gesture.config import *

import os, re
import hdf5storage
from common_dl import set_random_seeds


from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
if len(sys.argv)>3:
    sid = int(float(sys.argv[1]))
    model_name = sys.argv[2]
    fs = int(float(sys.argv[3]))
    wind = int(float(sys.argv[4]))
    stride = int(float(sys.argv[5]))
    try:
        depth=int(float(sys.argv[6]))
        print("Depth: "+ str(depth))
    except IndexError:
        pass
else: # debug in IDE
    sid=10
    fs=1000
    wind = 500
    stride = 500
    model_name='deepnet_varyBlocks'
class_number=5
#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

model_path=data_dir + '/training_result/model_pth/'+str(sid)+'/'
result_path=data_dir+'training_result/deepLearning/'+str(sid)+'/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

data_path = data_dir+'preprocessing/'+'P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(data_path)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
# total channel = channelNum + 4(2*emg + 1*trigger_indexes + 1*emg_trigger)
data=np.concatenate((data[0,0],data[0,1]),0) # (1052092, 212)
del mat
data=data.transpose()# (212, 1052092)

# standardization
# no effect. why?
if 1==0: #(n_samples, n_features)
    chn_data=data[-4:,:]
    data=data[:-4,:]
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform(data)
    data=np.concatenate((data,chn_data),axis=0)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data, info)


# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim_trigger')
events1 = mne.find_events(raw, stim_channel='stim_emg')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
raw=raw.pick(list(np.arange(162,168)))
epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None).load_data().resample(500)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1'].get_data()
epoch3=epochs['2'].get_data()
epoch4=epochs['3'].get_data()
epoch5=epochs['4'].get_data()
list_of_epoch=np.concatenate((epoch1,epoch2,epoch3,epoch4,epoch5),axis=0) # (100, 208, 4001)

list_of_labes=[]
for i in range(5):
    trialNum=epoch1.shape[0]
    label=[[i+1,]*trialNum]
    list_of_labes.append(label)
list_of_labes=np.squeeze(np.asarray(list_of_labes))
list_of_labes=np.squeeze(list_of_labes.reshape((1,-1))) # (100,)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(list_of_epoch,list_of_labes,test_size=0.3,random_state=222) # (70, 208, 4001)

#%%
fs = 500
n_w = 2
n_features = 4

labels_name = {}
labels_name[0] = '0'
labels_name[1] = '1'
labels_name[2] = '2'
labels_name[3] = '3'
labels_name[4] = '4'
labels_name[5] = '5'

print_var = True

accuracy_list = []
accuracy_matrix = []

# idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [1, 2, 3, 6, 7, 8]
idx_list = [4]
repetition = 1

#%%
debug=False
for rep in range(repetition):
    for idx in idx_list:
    # for idx in range(1, 10):
        print('Subject n.', str(idx))
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Train
        if not 'dubug':
            # Path for 4 classes dataset
            path_train = '/Volumes/Samsung_T5/data/BCI_competition/BCICIV_2a_gdf/Train'
            path_train_label = '/Volumes/Samsung_T5/data/BCI_competition/true_labels/A0'+str(idx) + 'E.mat'

            data, event_matrix = loadDatasetD2(path_train, idx)

            trials, labels = computeTrialD2(data, event_matrix, 250, remove_corrupt = False) # (192, 22, 1000),labels:[1,2,3,4,5]
            labels_1 = np.squeeze(loadmat(path_train_label)['classlabel'])
            labels_2 = loadTrueLabel(path_train_label)
            trials_dict = createTrialsDictD2(trials, labels, labels_name)
        else:
            trials_dict = createTrialsDictD2(X_train, y_train, labels_name) #
        #mybands=np.array([1, 4, 8, 13, 30, 50, 75, 100, 125, 150, 175])
        mybands = np.array([[1,4],[4,8],[8,13],[13,30],[60,75],[75,95],[105,125],[125,145],[155,195]])
        #svc_clf = LinearSVC(random_state=0, tol=1e-5)
        #clf = CalibratedClassifierCV(svc_clf)

        #FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, freqs_band=mybands, classifier=clf, print_var = print_var)
        FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, freqs_band=mybands,classifier =None,  print_var = print_var)
        #SVC(kernel = 'rbf', probability = True),
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Test set
        if not 'dubug':
            path_test = '/Volumes/Samsung_T5/data/BCI_competition/BCICIV_2a_gdf/Test'
            path_test_label = '/Volumes/Samsung_T5/data/BCI_competition/true_labels/A04E.mat'

            data_test, event_matrix_test = loadDatasetD2(path_test, idx)
            trials_test, labels_test = computeTrialD2(data_test, event_matrix_test, fs)
            data_test = -data_test

            labels_true_value_1 = np.squeeze(loadmat(path_test_label)['classlabel'])
            labels_predict_value = FBCSP_multi_clf.evaluateTrial(trials_test)

        # test
        else:
            labels_true_value_1=y_val
            labels_predict_value = FBCSP_multi_clf.evaluateTrial(X_val)
        
        a1 = FBCSP_multi_clf.pred_label_array
        a2 = FBCSP_multi_clf.pred_prob_array
        a3 = FBCSP_multi_clf.pred_prob_list
        
        # Percentage of correct prediction
        correct_prediction_1 = labels_predict_value[labels_predict_value == labels_true_value_1]
        perc_correct_1 = len(correct_prediction_1)/len(labels_true_value_1)
        accuracy_list.append(perc_correct_1)

        
        print('\nPercentage of correct prediction: ', perc_correct_1)
        print("# # # # # # # # # # # # # # # # # # # # #\n")
        
    accuracy_matrix.append(accuracy_list)
