'''
calculate correlation between ERS/ERD and task status using fixed low and high frequency range.
'''
import sys
import socket
from scipy.stats import spearmanr

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
import numpy as np
import matplotlib.pyplot as plt
from gesture.config import *
from gesture.preprocess.utils import *

savefile=data_dir+'tfAnalysis/ERSD_activation.npy'
filename=data_dir+'info/Info.npy'
info=np.load(filename,allow_pickle=True)
sids=info[:,0]

decimated_fs=250
fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
erdf=[5,20] #[5,20] as index in tfplot = [7,22] in real frequency
ersf=[40,100] #=[42,102]
onset=3.5-0.5 # after decimated in tf analysis. ERS/D are 0.5s ahead of movement.
baseline_num=int(onset*decimated_fs)
corr_sid={}
for sid in sids:

    tf_dir=data_dir + '/tfAnalysis/P'+str(sid)+'/'
    tf=tf_dir+'tf_data.npy'
    tf=np.load(tf) #(chnNum,time,frequency)
    chnNum=tf.shape[0]
    lent=tf.shape[2]

    tmp=[[1]*baseline_num,[-1]*(lent-baseline_num-int(0.5*decimated_fs))]
    erd_shape=np.concatenate(tmp)
    tmp=[[-1]*baseline_num,[1]*(lent-baseline_num-int(0.5*decimated_fs))]
    ers_shape=np.concatenate(tmp)

    perm_num=2500
    corr=np.zeros((2,chnNum))
    for c in range(chnNum):
        tmp1 = np.mean(tf[c,erdf[0]:erdf[1],:-int(0.5*decimated_fs)],axis=0)
        corr[0,c], _ = spearmanr(tmp1, erd_shape)

        tmp2 = np.mean(tf[c,ersf[0]:ersf[1],:-int(0.5*decimated_fs)], axis=0)
        corr[1,c], _ = spearmanr(tmp2, ers_shape)


    # get 10 largest correlation
    erd_index = corr[0,:].argsort()[-10:] #[::-1]
    ers_index = corr[1,:].argsort()[-10:]
    comm_chn=np.intersect1d(erd_index,ers_index)
    corr_sid[str(sid)]=corr
np.save(savefile, corr_sid) # corr_sid: dict with key=sid. ersd['sid'].shape=(2,channel number)
#read_dictionary = np.load(savefile,allow_pickle='TRUE').item()












