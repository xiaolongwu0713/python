from datetime import datetime
import sys,os
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import dateutil,pytz
import numpy as np
from dotmap import DotMap
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import common_dl
from gesture.DA.timegan.timegan import timegan
from gesture.DA.utils import sine_data_generation
from common_dl import *
from comm_utils import *
from torch.utils.data import DataLoader
from gesture.DA.VAE.VAE import SEEG_CNN_VAE, loss_fn, gen_data_vae, vae
from gesture.DA.GAN.DCGAN import dcgan
from gesture.DA.GAN.WGAN_GP import wgan_gp
from gesture.channel_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel, \
    get_selected_channel_stg
from gesture.channel_selection.mannal_selection import mannual_selection
from gesture.utils import *
from gesture.config import *
from gesture.DA import cfg_cmd
from gesture.DA.GAN.gan import gan

from common_dl import count_parameters


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

args = cfg_cmd.parse_args()
sid,fs,wind,stride,gen_method,continuous,epochs,selected_channels=args.sid,args.fs,args.wind,args.stride,args.gen_method,args.continuous,args.epochs,args.selected_channels
latent_dims=args.latent_dims
#print(str(sid)+','+str(fs)+','+str(wind)+','+str(stride)+','+gen_method+','+continuous+','+str(epochs)+','+str(selected_channels))

class_number = 5
if selected_channels:
    selected_channels, acc = get_selected_channel_gumbel(args.sid, 10) # 10 selected channels
else:
    selected_channels=None

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
if gen_method=='DCGAN' or gen_method=='CWGANGP' or gen_method=='CWGAN':
    norm_method='std' # minmax
else:
    norm_method='std' # minmax/std


def train(sid,continuous,train_loader,chn_num,save_gen_dir):
    now = datetime.now(pytz.timezone('Asia/Shanghai')) #dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    log_path = data_dir + 'DA/' + gen_method + '/' + str(sid)  + '/' + timestamp + '/'
    writer = SummaryWriter(log_path)
    experiment_description = "Generator: SEEG_CNN_Generator5 + Discriminator : the origianl SSVEP GAN models.\n"
    experiment_description = continuous + ": " + experiment_description
    writer.add_text('Experiment description', experiment_description, 0)
    print('Log path: ' + log_path + '.')

    if gen_method == 'VAE':
        print("Class" + str(classi) + ":Generate data using " + gen_method + ".")
        vae(chn_num, class_number, wind, scaler, train_loader, epochs, save_gen_dir, classi)
    elif gen_method == 'DCGAN':
        print("Class" + str(classi) + ":Generate data using " + gen_method + ".")
        dcgan(writer, sid, chn_num, class_number, wind, scaler, train_loader, epochs, save_gen_dir, result_dir,
              classi)
    elif gen_method == 'WGANGP':
        print("Class" + str(classi) + ":Generate data using " + gen_method + ".")
        wgan_gp(writer, sid, continuous, chn_num, class_number, wind, scaler, norm_method, train_loader, epochs,
                save_gen_dir, result_dir, classi, total_trials)
    elif gen_method in ['CWGANGP' ,'CDCGAN']:
        print("Generate data using " + gen_method + ".")
        gan(gen_method,writer, sid, continuous, chn_num, class_number, wind, train_loader, epochs, result_dir,latent_dims)
    elif gen_method == 'timeGAN':
        print("Class" + str(classi) + ":Generate data using " + gen_method + ".")
        args = DotMap()
        args.test = True  # test on the sine signal generation
        timegan(sid, train_loader, save_gen_dir, result_dir, classi, scaler, args)

batch_size = 32
test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid,fs,selected_channels=selected_channels,scaler=norm_method)
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride)
total_trials=X_train.shape[0]+X_val.shape[0]+X_test.shape[0]

if gen_method in ['CWGANGP','CDCGAN']:
    X_train=np.concatenate((X_train, X_val), axis=0)
    y_train=np.concatenate((y_train,y_val))
    train_set = myDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    one_batch = next(iter(train_loader))[0]  # torch.Size([32, 208, 500])
    chn_num = one_batch.shape[1]
    train(sid,continuous,train_loader,chn_num,save_gen_dir)

else:
    ## get different class data ##
    labels=np.array([i[0] for i in y_train.tolist()])
    X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (576, 208, 500), (576,)
    X_train_class1, labels1 = X_train[labels==1,:,:], labels[labels==1]
    X_train_class2, labels2 = X_train[labels==2,:,:], labels[labels==2]
    X_train_class3, labels3 = X_train[labels==3,:,:], labels[labels==3]
    X_train_class4, labels4 = X_train[labels==4,:,:], labels[labels==4]

    # test on one class
    class_number=5
    for classi in range(class_number): # [1,2,3,4]: #range(class_number):
        print('############### class '+str(classi)+ ' #############')
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
        # X_train: (304, 190, 500);y_train: (304,)
        test_sine=False
        if test_sine==True:
            X_train=sine_data_generation(304,500,208)
            X_train=np.asarray(X_train).transpose(0,2,1)
            y_train=[1,]*304

        if gen_method=='timeGAN':
            train_set = myDataset_timeGan(X_train, y_train)
            val_set = myDataset_timeGan(X_val, y_val)
            test_set = myDataset_timeGan(X_test, y_test)
            collate_fn=common_dl.collate_fn # swap dimensions
        else:
            train_set=myDataset(X_train,y_train)
            val_set=myDataset(X_val,y_val)
            test_set=myDataset(X_test,y_test)
            collate_fn=None
        batch_size = 32 # for all sids: len(train_loader)=10, means total 10 batchs in one training epoch
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False,collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False,collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False,collate_fn=collate_fn)

        one_batch=next(iter(train_loader))[0] # torch.Size([32, 208, 500])
        chn_num=one_batch.shape[1]
        train()

