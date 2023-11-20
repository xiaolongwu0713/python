'''
To slow: 1 hours for 1 channel.
'''

import sys
import socket

import scipy

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
import numpy as np
import matplotlib.pyplot as plt
from gesture.config import *
from gesture.preprocess.utils import *

sid=2
decimated_fs=250
tf_dir=data_dir + '/tfAnalysis/P'+str(sid)+'/'
tf=tf_dir+'tf_data.npy'
tf=np.load(tf)
chnNum=tf.shape[0]
lent=tf.shape[2]

fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148

erd_wind=5
ers_wind=5
erd_end_f=40
ers_start_f=40
erd_start_f_index = getIndex(fMin, fMax, fstep, fMin)
erd_end_f_index = getIndex(fMin, fMax, fstep, erd_end_f)  # 30
ers_start_f_index = getIndex(fMin, fMax, fstep, ers_start_f)  # 50
ers_end_f_index = getIndex(fMin, fMax, fstep, fMax)

erd_num=erd_end_f_index-erd_start_f_index+1-erd_wind+1
ers_num=ers_end_f_index-ers_start_f_index+1-ers_wind+1
erd_mean=np.zeros((chnNum,erd_num))
ers_mean=np.zeros((chnNum,ers_num))
onset=3.5 # after decimated in tf analysis
baseline_num=int(3.5*decimated_fs)
tmp=[[1]*baseline_num,[-1]*(lent-baseline_num)]
erd_shape=np.concatenate(tmp)
tmp=[[-1]*baseline_num,[1]*(lent-baseline_num)]
ers_shape=np.concatenate(tmp)

from scipy.stats import spearmanr
import itertools
import random
from scipy.stats import norm

perm_num=2500
best_erd = np.zeros(chnNum,)
best_ers = np.zeros(chnNum,)
best_p_value_erd = np.zeros(chnNum,)
best_p_value_ers = np.zeros(chnNum,)
for c in range(chnNum):
    print("channel "+str(c)+".")
    for i in range(erd_num):
        tmp=np.mean(tf[c,erd_start_f_index+i:erd_start_f_index+i+erd_wind,:],axis=0)
        corr, _ = spearmanr(tmp, erd_shape)
        r_pdf=np.zeros((perm_num))
        for j in range(perm_num):
            index_list=list(range(len(tmp)))
            random.shuffle(index_list)
            tmp_perm =tmp[index_list]
            r_pdf[j], _ = spearmanr(tmp_perm, erd_shape)
            # test r_pdf as a norm distribution:
            # plt.hist(r_pdf)
        p_value = 2 * scipy.stats.norm.cdf(-abs(corr), np.mean(r_pdf), np.std(r_pdf))
        if i==0:
            best_p_value_erd[c]=p_value
            best_erd[c] = i
        if p_value < best_p_value_erd[c]:
            best_p_value_erd[c]=p_value
            best_erd[c]=i

    for i in range(ers_num):
        tmp = np.mean(tf[c, ers_start_f_index + i:ers_start_f_index + i + ers_wind, :], axis=0)
        corr, _ = spearmanr(tmp, ers_shape)
        r_pdf = np.zeros((perm_num))
        for j in range(perm_num):
            index_list = list(range(len(tmp)))
            random.shuffle(index_list)
            tmp_perm = tmp[index_list]
            r_pdf[j], _ = spearmanr(tmp_perm, ers_shape)
        p_value = 2 * scipy.stats.norm.cdf(-abs(corr), np.mean(r_pdf), np.std(r_pdf))
        if i==0:
            best_p_value_ers[c]=p_value
            best_ers[c] = i
        if p_value < best_p_value_ers[c]:
            best_p_value_ers[c] = p_value
            best_ers[c] = i

erd_freq = np.zeros(chnNum,)
ers_freq = np.zeros(chnNum,)
for c in range(chnNum):
    erd_freq[c]=erd_start_f_index+best_erd[c]
    ers_freq[c]=ers_start_f_index+best_ers[c]

a=np.argpartition(np.min(erd_mean,axis=1),10)
b=np.argpartition(np.max(ers_mean,axis=1),-10)
min_erd=np.argmin(erd_mean,axis=1)
max_ers=np.argmax(ers_mean,axis=1)


erd_wind_avg = np.convolve(tf[:,erd_start_f_index:erd_end_f_index], np.ones(erd_wind) / erd_wind, mode='valid')
ers_wind_avg = np.convolve(tf[:,ers_start_f_index:ers_end_f_index], np.ones(ers_wind) / ers_wind,mode='valid')



import numpy as np

# Sample from a normal distribution using numpy's random number generator
samples = np.random.normal(size=10000)

# Compute a histogram of the sample
bins = np.linspace(-1, 1, 1000)
histogram, bins = np.histogram(samples, bins=bins, density=True)

bin_centers = 0.5*(bins[1:] + bins[:-1])

# Compute the PDF on the bin centers from scipy distribution object
from scipy import stats
pdf = stats.norm.pdf(bin_centers)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, histogram, label="Histogram of samples")
plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()











