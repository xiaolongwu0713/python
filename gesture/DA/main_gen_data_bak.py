import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
from gesture.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from common_dl import *
from comm_utils import slide_epochs
from torch.utils.data import DataLoader
from gesture.DA.VAE.VAE import SEEG_CNN_VAE, loss_fn, gen_data_vae, vae
from gesture.DA.GAN.DCGAN import dcgan
from gesture.DA.GAN.WGAN_GP2 import wgan_gp
from gesture.feature_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel, \
    get_selected_channel_stg
from gesture.feature_selection.mannal_selection import mannual_selection
from gesture.utils import *
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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


'''
run from cmd on workstation
'''
if running_from_CMD:
    if socket.gethostname() == 'workstation' or socket.gethostname() == 'DESKTOP-NP9A9VI':
        sid = int(float(sys.argv[1]))
        fs = int(float(sys.argv[2]))
        wind = int(float(sys.argv[3]))  # 500
        stride = int(float(sys.argv[4]))  # 200
        #task = sys.argv[5]  # 'gen_data'/'retrain_model'
        gen_method=sys.argv[5].upper() # 'VAE'/'DCGAN'/'WGAN_GP'
        continuous=sys.argv[6] # resume/fresh
        epochs=int(float(sys.argv[7]))
else:
    '''
    default setting
    '''
    sid = 41
    fs = 1000
    wind = 500
    stride = 200
    gen_method = 'WGAN_GP'  # VAE/GAN
    class_number = 5
    continuous = 'fresh'
    epochs = 200

save_gen_dir = data_dir + 'preprocessing/' + 'P' + str(sid)+'/'+gen_method+'/'
result_dir = result_dir + 'DA/' + gen_method+ '/'+ str(sid) + '/'
model_dir=result_dir

print_this="Python: Generate more data with " + gen_method+ "." if continuous=='fresh' else "Python: Resume generate more data with " + gen_method+ "."
print(print_this)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(save_gen_dir):
    os.makedirs(save_gen_dir)
print('Result dir: '+result_dir+'.')
print('Generate data in dir: '+save_gen_dir+'.')


# can try to use selected channels
test_epochs, val_epochs, train_epochs, scaler=read_data(sid,fs,scaler='minmax')
# X_train=[]
# y_train=[]
# X_val=[]
# y_val=[]
# X_test=[]
# y_test=[]
#
# for clas, epochi in enumerate(test_epochs):
#     Xi,y=slide_epochs(epochi,clas,wind, stride)
#     assert Xi.shape[0]==len(y)
#     X_test.append(Xi)
#     y_test.append(y)
# X_test=np.concatenate(X_test,axis=0) # (1300, 63, 500)
# y_test=np.asarray(y_test)
# y_test=np.reshape(y_test,(-1,1)) # (5, 270)
#
# for clas, epochi in enumerate(val_epochs):
#     Xi,y=slide_epochs(epochi,clas,wind, stride)
#     assert Xi.shape[0]==len(y)
#     X_val.append(Xi)
#     y_val.append(y)
# X_val=np.concatenate(X_val,axis=0) # (1300, 63, 500)
# y_val=np.asarray(y_val)
# y_val=np.reshape(y_val,(-1,1)) # (5, 270)
#
# for clas, epochi in enumerate(train_epochs):
#     Xi,y=slide_epochs(epochi,clas,wind, stride)
#     assert Xi.shape[0]==len(y)
#     X_train.append(Xi)
#     y_train.append(y)
# X_train=np.concatenate(X_train,axis=0) # (2880, 208, 500)
# y_train=np.asarray(y_train)
# y_train=np.reshape(y_train,(-1,1)) # (2880, 1)

X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride)

## get different class data ##
labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (576, 208, 500), (576,)
X_train_class1, labels1 = X_train[labels==1,:,:], labels[labels==1]
X_train_class2, labels2 = X_train[labels==2,:,:], labels[labels==2]
X_train_class3, labels3 = X_train[labels==3,:,:], labels[labels==3]
X_train_class4, labels4 = X_train[labels==4,:,:], labels[labels==4]

# test on one class
class_number=1
for classi in range(class_number):
    if classi==0:
        X_train, y_train = X_train_class0, labels0
    elif classi==1:
        X_train, y_train = X_train_class1, labels1
    elif classi==2:
        X_train, y_train = X_train_class2, labels2
    elif classi==3:
        X_train, y_train = X_train_class3, labels3
    elif classi==4:
        X_train, y_train = X_train_class4, labels4

    train_set=myDataset(X_train,y_train)
    val_set=myDataset(X_val,y_val)
    test_set=myDataset(X_test,y_test)

    batch_size = 8
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    one_window=next(iter(train_loader))[0] # torch.Size([32, 208, 500])
    chn_num = one_window.shape[1]
    wind_size=one_window.shape[2]

    if gen_method == 'VAE':
        print("Class"+str(classi)+":Generate data using "+gen_method+".")
        vae(chn_num, class_number, wind_size, scaler,train_loader, epochs,save_gen_dir,classi)
    elif gen_method == 'GAN':
        print("Class"+str(classi)+":Generate data using " + gen_method + ".")
        dcgan(chn_num, class_number, wind_size, scaler,train_loader, epochs,save_gen_dir,classi)
    elif gen_method == 'WGAN_GP':
        print("Class"+str(classi)+":Generate data using " + gen_method + ".")
        wgan_gp(sid,continuous, chn_num, class_number, wind_size, scaler, train_loader, epochs,save_gen_dir,result_dir,classi)







