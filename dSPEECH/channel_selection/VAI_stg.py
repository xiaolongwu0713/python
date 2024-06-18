'''
This script is to select channels best contribute to the VAI task;
'''

import mne
import numpy as np
from librosa import load
from tqdm import tqdm

from dSPEECH.config import *
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.deepmodel import deepnet
from gesture.models.selectionModels_gumble import selectionNet
from gesture.models.stg.stg import STG
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
sid=1 # 1/2
sf=1024
result_dir = data_dir + 'result/'+modality+str(sid)+'/VAI_channel_selection/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
## load epochs and sentences
filename=data_dir+'processed/'+modality+str(sid)+'/'+modality+str(sid)+'-epo.fif'
epochs=mne.read_epochs(filename)
filename2=data_dir+'processed/'+modality+str(sid)+'/sentences.npy'
sentences=np.load(filename2,allow_pickle=True)

# two events: 1 stands for 'TRIG[001]:1' and 2 stands for 'TRIG[001]:1:inserted'
epoch1=epochs['1'].get_data() # (99, 150, 15361)
epoch2=epochs['2'].get_data() # (4, 150, 15361)
epochs=np.concatenate((epoch1,epoch2),axis=0) # (103 trials, 150 channels, 15361 time)
# inspect the audio shows that: 500ms silence at the beginning and end of the sentence
# afile=mydriver+'/matlab/paradigms/speech_Southmead/audio/original/15_second_wavs/1.wav'
# audio, sf = load(afile,sr=None,mono=False)
# fig,ax=plt.subplots()
# plt.show()
# ax.plot(audio)
tmp=np.zeros(epochs.shape)
scaler = StandardScaler()
for i in range(len(epochs)):
    tmp1 = scaler.fit_transform((epochs[i,:,:].transpose()))
    tmp[i,:,:]=tmp1.transpose()
epochs=tmp

transition=0.5
listen=epochs[:,:,int(transition*sf):int((5-transition)*sf)] # (100, 150, 4096)
speak=epochs[:,:,int((5+transition)*sf):int((10-transition)*sf)] # (100, 150, 4096)
image=epochs[:,:,int((10+transition)*sf):int((15-transition)*sf)] # (100, 150, 4096)

def train_test_split(data):
    trial_number=data.shape[0]
    trial_list = list(range(trial_number))
    train_n=int(0.6*trial_number)
    val_n = int(0.2 * trial_number)
    test_n = int(0.2 * trial_number)

    test_trails=random.sample(trial_list, test_n)
    trial_number_left=np.setdiff1d(trial_list,test_trails)

    val_trails = random.sample(trial_number_left.tolist(), val_n)
    train_trails = np.setdiff1d(trial_number_left, val_trails)
    return data[train_trails,:,:], data[val_trails,:,:], data[test_trails,:,:]

train_listen,val_listen,test_listen=train_test_split(listen)
train_speak,val_speak,test_speak=train_test_split(speak)
train_image,val_image,test_image=train_test_split(image)

class_number=3
test_lists=[test_listen,test_speak,test_image]
val_lists=[val_listen,val_speak,val_image]
train_lists=[train_listen,train_speak,train_image]
wind=200
stride=70
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_lists,val_lists,test_lists,wind,stride)

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 32 # larger batch_size slows the training
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190

# Extract number of chans and time steps from dataset
one_window=next(iter(train_loader))[0]
n_chans = one_window.shape[1]
input_window_samples=one_window.shape[2]

args_cuda = torch.cuda.is_available()
device = torch.device("cuda" if args_cuda else "cpu")

def two_stages_ascend(start_value,end_value,epochs,end_epoch):
    t = torch.FloatTensor(torch.arange(0.0,epochs))
    out=torch.zeros(epochs)
    out[:end_epoch]=start_value
    a=(end_value-start_value)/(epochs-end_epoch)
    b=end_value-epochs*a
    y=a*t[end_epoch:]+b
    out[end_epoch:]=y
    return out

epochs=200
start_to_shrink=20
lam_schedule=two_stages_ascend(0.00000001,0.5, epochs, start_to_shrink)
patients=5
task_meta={}
task_meta['task']='classification'
task_meta['result_dir']=result_dir
# freeze_onward=20; when to freeze the selection net???
lam=0.2 # larger lam, less chanelle selected; 0.2:26 channels selected; 0.1:35; 0.05: 42; 0.01: 40
model = STG(n_chans,class_number,wind, task_meta, freeze_onward=None, patients=patients, task_type='classification',batch_size=32, sigma=0.5, lam=lam, lam_schedule=lam_schedule, random_state=1, device=device)

result_prob, result_raw, train_loss, validate_acc=model.fit(train_loader, val_loader, epochs, print_interval=1, early_stop=True)
result_prob=np.asarray(result_prob)
result_raw=np.asarray(result_raw)
train_loss=np.asarray(train_loss)
validate_acc=np.asarray(validate_acc)
prob=model.get_gates(mode='prob')
raw=model.get_gates(mode='raw')
import matplotlib.pyplot as plt
#plt.plot(prob)
#plt.plot(raw)

# testing
truth, y_pred=model.predict(test_loader)
test_acc=sum(truth==y_pred)/truth.shape[0]
print('Testing accuracy: '+str(test_acc)+'.')

result={}
result['probs']=result_prob
result['raws']=result_raw
result['train_loss']=train_loss
result['validate_acc']=validate_acc
result['test_acc']=test_acc
filename=result_dir+'result'+str(sid)+'_stg_lam'+str(lam)+'.npy' #"result"+str(sid)+'_'+str(lam)
np.save(filename, result)
#load_back=np.load('result10.npy', allow_pickle=True).item()




