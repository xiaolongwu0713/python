import sys
import socket

import numpy as np

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long':  # Yoga
    sys.path.extend(['D:/mydrive/python/'])

from gesture.channel_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel, \
    get_selected_channel_stg
from gesture.utils import *
from gesture.config import *
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# if 'PYTHONPATH' in os.environ and 'PyCharm' in os.environ['PYTHONPATH']:
if os.environ.get('PYCHARM_HOSTED'):
    running_from_IDE = True
    running_from_CMD = False
    print("Running from IDE.")
else:
    running_from_CMD = True
    running_from_IDE = False
    print("Running from CMD.")

if running_from_CMD:  # run from cmd on workstation
    if socket.gethostname() == 'Long' or socket.gethostname() == 'DESKTOP-NP9A9VI':
        sid = int(float(sys.argv[1]))
        cv = int(float(sys.argv[2]))

else:  # run from IDE
    sid = 10
    cv = 1

fs = 1000
wind = 500
stride = 200
train_mode = 'DA'  # 'DA'/original/'selected_channels'
method = 'NI'  # 'NI'/'VAE'/'CWGANGP'/
gen_epochs = 200
result_dir='D:/tmp/python/gesture/DA/NI/sid' + str(sid) + '/cv'+str(cv)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print('Result Path: '+result_dir+'.')
class_number = 5

channel_num_selected = 10
selected_channels, acc = get_selected_channel_gumbel(sid, channel_num_selected)
#selected_channels=False
print("Python: Re-train with augmentated data generated by Noise Injection.")

scaler='std' # 'std'/None
#test_epochs, val_epochs, train_epochs, scaler=read_data(sid,fs,selected_channels=selected_channels,scaler=scaler)
test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid, fs, selected_channels=selected_channels,scaler='std',cv_idx=cv)

gen_data_all=False
X_train, y_train, X_val, y_val, X_test, y_test = windowed_data(train_epochs, val_epochs, test_epochs, wind, stride,
                                                               gen_data_all=gen_data_all, train_mode=train_mode,method=method)

print('Save gen data.')
std_scale=0.1 # 0.1: 0.75 how much noise added
train_gen=noise_injection_3d(X_train,std_scale)
for classi in range(5):
    classi_data=[x for x,y in zip(train_gen, y_train) if y==classi]
    classi_data=np.asarray(classi_data)
    filename=result_dir+'class'+str(classi)+'_cv'+str(cv)+'.npy'
    np.save(filename, classi_data)


