#savemat must use a str as key, not int

from scipy.io import savemat
import hdf5storage
import os, re
import matplotlib.pyplot as plt
from gesture.config import *


info_file=info_dir+'Info.npy'
info=np.load(info_file,allow_pickle=True)
good_channels={}
good_channels['good_channels']=[]
for sid in info[:,0]:
    loadPath = data_dir+'preprocessing'+'/P'+str(sid)+'/preprocessing1.mat'
    mat=hdf5storage.loadmat(loadPath)
    tmp=mat['good_channels']
    good_channels['good_channels'].append({'sid'+str(sid): np.squeeze(tmp)})
filename=meta_dir+"good_channels.mat"
savemat(filename, good_channels)





