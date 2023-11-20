'''
This script use a simple linear regression as a baseline;
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

def baseline(data_name=None,sid=1,session=1,test_shift=False,test_shifts=None,model_order=None, step_size=None,
             melbins=23,winL=0.05, frameshift=0.01,estimate_random_baseline=False,save_result=False):
    pass

from speech_Ruijin.seq2seq.opt import channel_numbers
if running_from_CMD: # run from cmd on workstation
    sid = int(float(sys.argv[1]))
else:
    sid = 3
channel_number=channel_numbers[sid-1]

from speech_Ruijin.baseline_linear_regression.opt import opt
model_order = opt['model_order']  # 4 # 4
step_size = opt['step_size']  # 4 # 5:0.82 1:0.76
winL = opt['winL']
frameshift = opt['frameshift']
melbins = 23
data_name = 'SingleWordProductionDutch'  # 'SingleWordProductionDutch'/'mydata'/'Huashan'

if computer=='mac':
    feat_path = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch/features'
    result_path = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch/results'
else:
    feat_path = r'H:\Long\data\SingleWordProductionDutch-iBIDS\features'
    result_path = r'H:\Long\data\SingleWordProductionDutch-iBIDS\results'
audiosr = 16000

nfolds = 5
kf = KFold(nfolds,shuffle=False)
est = LinearRegression(n_jobs=5)
pca = PCA()
numComps = 50

#Initialize empty matrices for correlation results, randomized contols and amount of explained variance
allRes = np.zeros((1,nfolds,melbins))
explainedVariance = np.zeros((1,nfolds))
numRands = 100
randomControl = np.zeros((1,numRands, melbins))

sids=[1,2,3,4,5,6,7,8,9,10]
#sids = [3,]

#Load the data
pt = 'sub-' + f"{sid:02d}"
print("Running for "+pt+'.')
result_dir = data_dir + 'baseline_LR_channel_selection/SingleWordProductionDutch/mel_' + str(melbins) + '/sid' + str(sid) + '/'
print('Result folder: ' + result_dir + '.')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

all_channels=list(range(channel_number))
from selected_channels import selected_channels
opt_channels=selected_channels[str(sid)] #[]
channel_tobe_selected=5
for i in range(channel_tobe_selected):
    text_file = result_dir + 'result.txt'
    with open(text_file, 'a') as f:
        f.write("########### select the "+str(i)+"th channel ###########")
        f.write('\n')
    channels_left=[i for i in all_channels if i not in opt_channels]
    mses = []
    corrs=[]
    for j,channel in enumerate(channels_left):
        use_channels=opt_channels.copy()
        use_channels.append(channel)
        print('Testing channels: ' +','.join(str(i) for i in use_channels))

        data,spectrogram=dataset(dataset_name=data_name,sid=sid,use_channels=use_channels,melbins=melbins,modelOrder=model_order,
                                 stepSize=step_size,winL=winL, frameshift=frameshift)

        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((1,spectrogram.shape[1]))
        train, test = next(kf.split(data))
        #Z-Normalize with mean and std from the training data
        mu=np.mean(data[train,:],axis=0)
        std=np.std(data[train,:],axis=0)
        trainData=(data[train,:]-mu)/std
        testData=(data[test,:]-mu)/std

        #Fit PCA to training data
        pca.fit(trainData)
        #Get percentage of explained variance by selected components
        explainedVariance =  np.sum(pca.explained_variance_ratio_[:numComps])
        #Tranform data into component space
        trainData=np.dot(trainData, pca.components_[:numComps,:].T)
        testData = np.dot(testData, pca.components_[:numComps,:].T)

        #Fit the regression model
        est.fit(trainData, spectrogram[train, :])
        #Predict the reconstructed spectrogram for the test data
        rec_spec[test, :] = est.predict(testData)

        #Evaluate reconstruction of this fold
        for specBin in range(spectrogram.shape[1]):
            if np.any(np.isnan(rec_spec)):
                print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
            r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
            rs[0,specBin] = r
        #break
        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))

        truth,pred,rs = rec_spec[test,:],spectrogram[test,:],rs

        #np.save(result_dir + 'pred.npy', pred)
        #np.save(result_dir + 'truth.npy', truth)
        #ax[0].imshow(truth[:, :].transpose(), aspect='auto')
        #ax[1].imshow(pred[:, :].transpose(), aspect='auto')
        #fig.savefig(result_dir + 'result.png')
        mse = mean_squared_error(truth[:3000, :], pred[:3000, :])
        corr = np.mean(rs)
        corrs.append(corr)
        #ax[0].clear()
        #ax[1].clear()
        #Save results in numpy arrays
        #np.save(os.path.join(result_path,'linearResults.npy'),allRes)
        #np.save(os.path.join(result_path,'randomResults.npy'),randomControl)
        #np.save(os.path.join(result_path,'explainedVariance.npy'),explainedVariance)

        text_file = result_dir + 'result.txt'
        with open(text_file, 'a') as f:
            f.write('Testing channels: ' + ','.join(str(i) for i in use_channels)+'. corr:' + format(corr, '.3f')+ ';mse:' + format(mse, '.3f'))
            f.write('\n')
        mses.append(mse)

    best = corrs.index(max(corrs))
    best_channel = channels_left[best]
    opt_channels.append(best_channel)
    with open(text_file, 'a') as f:
        f.write('selected channel:' + str(best_channel)+'; opt channel:'+','.join(str(i) for i in opt_channels))
        f.write('\n')

with open(text_file, 'a') as f:
    f.write('Final opt channel:'+','.join(str(i) for i in opt_channels))
    f.write('\n')
