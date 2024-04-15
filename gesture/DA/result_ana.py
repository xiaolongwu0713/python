'''
Jensen-Shannon Distance (JSD):
Cosine Similarity (CS):
'''
import argparse
import copy
import sys, os
import socket

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])

from scipy.spatial import distance
import numpy as np
from gesture.channel_selection.utils import get_selected_channel_gumbel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from gesture.DA.add_noise.noise_injection import noise_injection_3d
from gesture.config import tmp_dir, time_stamps
from gesture.utils import read_data_split_function, windowed_data
from pre_all import running_from_CMD, running_from_IDE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', default=10, type=int)
    parser.add_argument('--cv', type=int)
    #parser.add_argument('--method',type=str,default=False)
    opt = parser.parse_args()
    return opt
if running_from_CMD:
    args = parse_args()
    sid = args.sid
    cv=args.cv
    #method=args.method
elif running_from_IDE:
    #args = parse_args()
    sid = 10
    cv=1
    #method='cTGAN' #'VAE' #'cTGAN'

fs=1000
channel_num_selected = 10
selected_channels, acc = get_selected_channel_gumbel(sid, channel_num_selected)
wind=500
stride=200
norm_method='std'

test_epochs, val_epochs, train_epochs, scaler = read_data_split_function(sid, fs, selected_channels=selected_channels,
                                                                         scaler=norm_method,cv_idx=cv)
X_train, y_train, X_val, y_val, X_test, y_test = windowed_data(train_epochs, val_epochs, test_epochs, wind, stride)
total_trials = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (304, 10, 500)


for method in ['cTGAN','CWGANGP','VAE','NI']:
    method='NI'
    gen = []
    if method=='cTGAN':
        timestamp_cv = time_stamps[0][cv - 1]  # ['2024_04_13_15_59_58',] # 5 time stamps
        prefix = tmp_dir + 'DA/cTGAN/sid' + str(sid) + '/cv' + str(cv) + '/' + timestamp_cv + '/'
        sample_path = prefix + 'Samples/'
        for i in range(5):
            filename = sample_path + 'class' + str(i) + '_cv' + str(cv) + '.npy'
            tmp = np.load(filename)
            gen.append(tmp)
    elif method=='CWGANGP':
        timestamp_cv = time_stamps[1][cv - 1]  # ['2024_04_13_15_59_58',] # 5 time stamps
        prefix = tmp_dir + 'DA/CWGANGP/sid' + str(sid) + '/cv' + str(cv) + '/' + timestamp_cv + '/'
        sample_path = prefix + 'Samples/'
        for i in range(5):
            filename = sample_path + 'class' + str(i) + '_cv' + str(cv) + '.npy'
            tmp = np.load(filename)
            gen.append(tmp)
    elif method=='VAE':
        timestamp_cv= time_stamps[2][cv-1]#['2024_04_13_15_59_58',] # 5 time stamps
        prefix = tmp_dir + 'DA/VAE/sid' + str(sid) + '/cv' + str(cv) + '/' + timestamp_cv + '/'
        sample_path = prefix + 'Samples/'
        for i in range(5):
            filename=sample_path+'class'+str(i)+'_cv'+str(cv)+'.npy'
            tmp=np.load(filename)
            gen.append(tmp)
    elif method=='NI':
        std_scale = 0.1  # 0.1: 0.75 how much noise added
        gen = noise_injection_3d(X_train, std_scale)
    if method!='NI':
        gen=np.concatenate(gen) # (304*5=1520, 10, 500)
    real=X_train

    # scale to [0,1] range
    gen1=np.ones(gen.shape)
    real1=np.ones(real.shape)
    for t in range(len(gen)):
        scaler = MinMaxScaler(feature_range=(0,1))
        gen1[t] = scaler.fit_transform((gen[t]))
    for t in range(len(real)):
        scaler = MinMaxScaler(feature_range=(0,1))
        real1[t] = scaler.fit_transform((real[t]))

    # Jensen-Shannon Distance (JSD)
    jsd=[]
    for t in range(gen1.shape[0]):
        a=distance.jensenshannon(gen1[t,:,:], real1[t,:,:], axis=1)
        jsd.append(a.mean())
    jsd_avg=sum(jsd)/len(jsd)
    print('sid:'+str(sid)+'/'+method+':JSD:'+str(jsd_avg)+'.')
    # cosine similarity
    cs=[]
    for t in range(gen1.shape[0]):
        for c in range(gen1.shape[1]):
            cs.append(cosine_similarity(gen1[t,c,:].reshape(1, -1), real1[t,c,:].reshape(1, -1)))
    cs=np.asarray(cs)
    cs_avg=cs.mean()
    print('sid:'+str(sid)+'/'+method+':CS:'+str(cs_avg)+'.')





