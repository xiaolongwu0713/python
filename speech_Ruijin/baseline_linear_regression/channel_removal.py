'''
This script use a simple linear regression as a baseline;
This script reconstruct spectrogram by consecutively removing the most informative channels (one by one).
'''

import os
import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['C:/Users/xiaowu/mydrive/python/'])

import scipy
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pre_all import computer, running_from_CMD
import example_speech.SingleWordProductionDutch.reconstructWave as rW
import example_speech.SingleWordProductionDutch.MelFilterBank as mel
from speech_Ruijin.baseline_linear_regression.extract_features import dataset, extractHG, stackFeatures, extractMelSpecs
from speech_Ruijin.config import data_dir, extra_EEG_extracting
from speech_Ruijin.seq2seq.opt import channel_numbers
from speech_Ruijin.baseline_linear_regression.opt import opt

if running_from_CMD: # run from cmd on workstation
    sid = int(float(sys.argv[1]))
else:
    sid = 3
channel_number=channel_numbers[sid-1]
channel_tobe_removal=15
all_channels=list(range(channel_number))
from selected_channels_locations import selected_channels
opt_channels=selected_channels[str(sid)] #[]

model_order = opt['model_order']  # 4 # 4
step_size = opt['step_size']  # 4 # 5:0.82 1:0.76
winL = opt['winL']
frameshift = opt['frameshift']
melbins = 23
data_name = 'SingleWordProductionDutch'  # 'SingleWordProductionDutch'/'mydata'/'Huashan'

nfolds = 5
kf = KFold(nfolds,shuffle=False)
est = LinearRegression(n_jobs=5)
pca = PCA()
numComps = 50

pt = 'sub-' + f"{sid:02d}"
print(" ########  Running for "+pt+'. ##########')
result_dir = data_dir + 'baseline_LR_channel_removal/SingleWordProductionDutch/mel_' + str(melbins) + '/sid' + str(sid) + '/'
print('Result folder: ' + result_dir + '.')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_file = result_dir + 'result.txt'
with open(result_file, 'a') as f:
    f.write('model order:'+str(model_order)+'; step size:'+str(step_size)+';winL:'+str(winL)+';frameshift:'+str(frameshift)+'.')
    f.write('\n')

removed_sofar=[]
corrs=[]
for j in range(channel_tobe_removal+1):
    if j==0:
        channels_left = all_channels
        print('Total channel number:'+str(len(all_channels))+'Using all channels:'+str(len(channels_left)))
        with open(result_file, 'a') as f:
            f.write('########### Using all channels. Channel number:'+str(len(channels_left))+' ###########')
            f.write('\n')

    else:
        i=j-1
        ichannel=opt_channels[i]
        removed_sofar.append(ichannel)
        with open(result_file, 'a') as f:
            f.write("########### Removing channel: "+str(ichannel)+" ###########")
            f.write('\n')
        channels_left=[i for i in all_channels if i not in removed_sofar]
        print('Testing channels: ' + ','.join(str(i) for i in channels_left)+' Channel number:'+str(len(channels_left))+'.')

    data, spectrogram = dataset(dataset_name=data_name, sid=sid, use_channels=channels_left, melbins=melbins,
                                modelOrder=model_order,stepSize=step_size, winL=winL, frameshift=frameshift)

    # Initialize an empty spectrogram to save the reconstruction to
    rec_spec = np.zeros(spectrogram.shape)
    # Save the correlation coefficients for each fold
    rs = np.zeros((nfolds, spectrogram.shape[1]))
    for k, (train, test) in enumerate(kf.split(data)):
        # Z-Normalize with mean and std from the training data
        mu = np.mean(data[train, :], axis=0)
        std = np.std(data[train, :], axis=0)
        trainData = (data[train, :] - mu) / std
        testData = (data[test, :] - mu) / std

        # Fit PCA to training data
        pca.fit(trainData)
        # Tranform data into component space
        trainData = np.dot(trainData, pca.components_[:numComps, :].T)
        testData = np.dot(testData, pca.components_[:numComps, :].T)

        # Fit the regression model
        est.fit(trainData, spectrogram[train, :])
        # Predict the reconstructed spectrogram for the test data
        rec_spec[test, :] = est.predict(testData)

        # Evaluate reconstruction of this fold
        for specBin in range(spectrogram.shape[1]):
            if np.any(np.isnan(rec_spec)):
                print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
            r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
            rs[k, specBin] = r
        # break
    mean_corr=np.mean(rs)
    corrs.append(mean_corr)
    print('Mean correlation: %f.' % (mean_corr))

    with open(result_file, 'a') as f:
        f.write('Mean corr: ' + format(mean_corr,'.3f')+'.')
        f.write('\n')
