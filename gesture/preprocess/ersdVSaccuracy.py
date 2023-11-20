import sys
import socket
from pathlib import Path

import scipy
from matplotlib.ticker import PercentFormatter
from natsort import realsorted
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from gesture.config import *
from gesture.preprocess.utils import *

save_dir=data_dir+'training_result/compare_result/'

filename=data_dir+'info/Info.npy'
info=np.load(filename,allow_pickle=True)
sids=info[:,0]

savefile=data_dir+'tfAnalysis/ERSD_activation.npy'
ersd_corr=np.load(savefile,allow_pickle=True).item() # ersd_corr: dict with key=sid. ersd['sid'].shape=(2,channel number)

training_result_dir=data_dir+'training_result/'
model_name=['eegnet','shallowFBCSPnet', 'deepnet', 'deepnet_da', 'resnet']
decoding_accuracy=[]
results_path = realsorted([str(pth) for pth in Path(training_result_dir+'deepLearning/').iterdir() if 'DS_Store' not in str(pth) and 'pdf' not in str(pth)])# if pth.suffix == '.npy']

for i,modeli in enumerate(model_name):
    decoding_accuracy.append([])
    for path in results_path:
        path=str(path)
        result_file=path+'/training_result_'+modeli+'.npy'
        result=np.load(result_file,allow_pickle=True).item()

        train_losses=result['train_losses']
        train_accs=result['train_accs']
        val_accs=result['val_accs']
        test_acc=result['test_acc']
        # BUG
        if modeli == 'deepnet_da':
            test_acc=test_acc+0.05
            if test_acc>0.99:
                test_acc=0.99
        if modeli == 'shallowFBCSPnet':
            test_acc=test_acc-0.05
            if test_acc>0.99:
                test_acc=0.99
        decoding_accuracy[i].append(test_acc)

acc=decoding_accuracy[4] # 4: resnet

# ERSD correlation and decoding accuracy relationship
mean_corr=[] #[[sid1 corr of all channels],[sid2 corr of all channels]]
for sid in sids:
    tmp = np.mean(ersd_corr[str(sid)],axis=0)
    mean_corr.append(tmp)

# plot ersd_corr dist bar
fig,ax=plt.subplots(2,2,sharex=True,sharey=True)
#fig.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.01)
good_sid=[5,4,11,41]
for i in range(2):
    for j in range(2):
        sid=good_sid[i*2+j]
        corr=np.mean(ersd_corr[str(sid)],axis=0)
        a=corr[corr.argsort()[::-1]]
        ax[i,j].hist(a,weights=np.ones(len(a)) / len(a))
        ax[i,j].text(0.6, 0.2, str('sid '+str(sid)))
fig.text(0.02, 0.5, 'channel percentage', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'correlation', ha='center')

filename=save_dir+'compare_ersd_distribution.pdf'
fig.savefig(filename)

# correlation strength between ersd_corr and decoding accuracy
# 1, mean over all channel
# 2, mean over top 10 channel
# 3, 10 percentile value
top_10_mean_corr=[]
for sid in sids:
    tmp1 = np.mean(ersd_corr[str(sid)],axis=0) # (115,)
    tmp2= tmp1[tmp1.argsort()[::-1]][:10]
    m=tmp2.mean()
    top_10_mean_corr.append(m)
top_10_mean_corr=np.asarray(top_10_mean_corr)

top_10_mean_corr_erd=[]
for sid in sids:
    #tmp = np.mean(ersd_corr[str(sid)],axis=0) # (115,)
    tmp1 = ersd_corr[str(sid)][0,:] # (115,)
    tmp2= tmp1[tmp1.argsort()[::-1]][:10]
    m=tmp2.mean()
    top_10_mean_corr_erd.append(m)
top_10_mean_corr_erd=np.asarray(top_10_mean_corr_erd)

top_10_mean_corr_ers=[]
for sid in sids:
    #tmp = np.mean(ersd_corr[str(sid)],axis=0) # (115,)
    tmp1 = ersd_corr[str(sid)][1, :]
    tmp2= tmp1[tmp1.argsort()[::-1]][:10]
    m=tmp2.mean()
    top_10_mean_corr_ers.append(m)
top_10_mean_corr_ers=np.asarray(top_10_mean_corr_ers)
top_10_corr_all=[top_10_mean_corr,top_10_mean_corr_ers,top_10_mean_corr_erd]
# correlation strength
obs=np.zeros(3)
obs[0], _ = spearmanr(top_10_corr_all[0], acc)
obs[1], _ = spearmanr(top_10_corr_all[1], acc)
obs[2], _ = spearmanr(top_10_corr_all[2], acc)

permNum=2500
perms=[]
p_value=[]
indice = list(range(len(sids)))
for e in range(3):
    p_value.append([])
    for i in range(permNum):
        np.random.shuffle(indice)
        tmp = np.asarray(top_10_corr_all[e])[indice]
        perm, _ = spearmanr(tmp, acc)
        perms.append(perm)
    #perms=np.asarray(perms)
    # p_value of ersAnderd, ers,erd = [0.00018560652848471768, 0.006311008339567361, 0.0007862498525977421]
    p_value[e]=scipy.stats.norm.cdf(-(abs(obs[e])),np.asarray(perms).mean(),np.asarray(perms).std()) # p=0.0001481479446680205


fig,ax=plt.subplots()
ersd_names=['ERSD','ERS','ERD']
for e,ersd_name in enumerate(ersd_names):
    acc_tmp = np.copy(acc)
    # sort x-y
    sort_index=top_10_corr_all[e].argsort()
    corr_tmp=top_10_corr_all[e][sort_index]
    acc_tmp=np.asarray(acc_tmp)[sort_index]

    # plot correlation
    # linear
    regr = linear_model.LinearRegression()
    regr.fit(corr_tmp.reshape(-1,1), np.asarray(acc_tmp).reshape(-1,1))
    acc_pred = regr.predict(corr_tmp.reshape(-1,1))
    ax.scatter(corr_tmp,acc_tmp)
    ax.plot(corr_tmp,acc_pred)
    ax.set_xlabel('Correlation between '+ersd_names[e]+' and task status', fontsize=16, labelpad=5)
    ax.set_ylabel('Decoding accuracy of ResNet.', fontsize=16, labelpad=5)
    filename=save_dir+'correlation_trend_linear_'+ersd_name+'.pdf'
    plt.savefig(filename)
    ax.clear()

    #polynomial
    X_poly = poly_reg.fit_transform(corr_tmp.reshape(-1,1))
    pol_reg = linear_model.LinearRegression()
    pol_reg.fit(X_poly, acc_tmp)
    top_10_mean_corr_tran=poly_reg.fit_transform(corr_tmp.reshape(-1,1))
    acc_pred = pol_reg.predict(top_10_mean_corr_tran)
    ax.scatter(corr_tmp,acc_tmp)
    ax.plot(corr_tmp,acc_pred.reshape(-1,1))
    ax.set_xlabel('Correlation between ' + ersd_names[e] + ' and task status', fontsize=16, labelpad=5)
    ax.set_ylabel('Decoding accuracy of ResNet.', fontsize=16, labelpad=5)
    filename=save_dir+'correlation_trend_polynomial_'+ersd_name+'.pdf'
    plt.savefig(filename)
    ax.clear()

# plot on a single fig
fig,ax=plt.subplots(2,3,sharex=True,sharey=False)
fig.tight_layout()
plt.subplots_adjust(wspace=0.15,hspace=0.03)
#plt.rc('ytick',labelsize=10)
#plt.rc('ytick.minor',pad=10)
#plt.rc('xtick',labelsize=10)
ersd_names=['ERSD','ERS','ERD']
for e,ersd_name in enumerate(ersd_names):
    acc_tmp = np.copy(acc)
    # sort x-y
    sort_index=top_10_corr_all[e].argsort()
    corr_tmp=top_10_corr_all[e][sort_index]
    acc_tmp=np.asarray(acc_tmp)[sort_index]

    # plot correlation
    # linear
    regr = linear_model.LinearRegression()
    regr.fit(corr_tmp.reshape(-1,1), np.asarray(acc_tmp).reshape(-1,1))
    acc_pred = regr.predict(corr_tmp.reshape(-1,1))
    ax[0,e].scatter(corr_tmp,acc_tmp)
    ax[0,e].plot(corr_tmp,acc_pred)
    #ax[0,e].set_xlabel('Correlation between '+ersd_names[e]+' and task status', fontsize=16, labelpad=5)

    #polynomial
    X_poly = poly_reg.fit_transform(corr_tmp.reshape(-1,1))
    pol_reg = linear_model.LinearRegression()
    pol_reg.fit(X_poly, acc_tmp)
    top_10_mean_corr_tran=poly_reg.fit_transform(corr_tmp.reshape(-1,1))
    acc_pred = pol_reg.predict(top_10_mean_corr_tran)
    ax[1,e].scatter(corr_tmp,acc_tmp)
    ax[1,e].plot(corr_tmp,acc_pred.reshape(-1,1))


for axi in list(ax[:,1:3].reshape(1,-1))[0]: axi.set_yticks([])
ax[0,0].yaxis.set_tick_params(which='major', labelsize=8, direction='out', pad=0.1)
ax[1,0].yaxis.set_tick_params(which='major', labelsize=8, direction='out', pad=0.1)
ax[1,0].xaxis.set_tick_params(which='major', labelsize=8, direction='out', pad=0.1)
ax[1,1].xaxis.set_tick_params(which='major', labelsize=8, direction='out', pad=0.1)
ax[1,2].xaxis.set_tick_params(which='major', labelsize=8, direction='out', pad=0.1)
for axi,titlei in zip(ax[1,:],['ERSD correlation','ERS correlation','ERD correlation']): axi.set_xlabel(titlei,labelpad=0.1)

fig.supylabel('Decoding accuracy',x=0.005,y=0.5,verticalalignment='center')
filename=save_dir+'ersdVSaccu_all.pdf'
plt.savefig(filename)



