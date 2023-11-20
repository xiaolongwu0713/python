import logging
import os
import time
from datetime import datetime
import pytz
import torch
import numpy as np
from torch.utils.data import Dataset

from gesture.channel_selection.utils import get_selected_channel_gumbel
from gesture.utils import read_data_split_function, windowed_data

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    #exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(pytz.timezone('Asia/Shanghai'))
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = root_dir  + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

class mydataset(Dataset):
    def __init__(self,args,norm='std',is_normalize=False,data_mode='Train', single_class=False, classi=0):

        self.norm=norm
        self.data_mode = data_mode
        self.classi = classi
        self.single_class = single_class
        self.is_normalize = is_normalize
        self.sid=args.sid
        self.fs=args.fs
        self.wind=args.wind
        self.chn=args.chn
        self.stride=args.stride
        if args.selected_channels==True:
            selected_channels, acc = get_selected_channel_gumbel(self.sid, args.chn)
        else:
            selected_channels=None

        # use the real data
        test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(self.sid,self.fs,scaler=self.norm,selected_channels=selected_channels)
        X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,self.wind,self.stride)
        # or uncomment below to use a dummy data set to test the program
        # X_train, y_train,  = np.random.rand(1520, 10, 500),np.random.rand(1520, 1),
        # X_val, y_val= np.random.rand(190, 10, 500), np.random.rand(190,1)
        # X_test, y_test=np.random.rand(190, 10, 500), np.random.rand(190,1)

        self.X_train=np.concatenate((X_train,X_val),axis=0)
        y_train=np.concatenate((y_train,y_val),axis=0)
        ## get different class data ##
        self.labels_train=np.array([i[0] for i in y_train.tolist()])
        # tested on the class 0
        if self.single_class:
            self.X_train = self.X_train[self.labels_train==self.classi,:,:] # (576, 208, 500), (576,)
            self.labels_train = self.labels_train[self.labels_train==self.classi]

        # format to (1524, 3, 1, 150), (100000, 1, 1, 187)
        self.X_train = self.X_train[:, :, np.newaxis, :]
    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.labels_train[idx]
