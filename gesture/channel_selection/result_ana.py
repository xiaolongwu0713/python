# sid10(208channels): [143, 144, 146, 147, 148, 149, 150, 151, 152, 167]
# sid41(190channels): [33, 163, 37, 166, 165, 19, 20, 26, 30, 63]

import hdf5storage
import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from gesture.channel_selection.utils import get_selected_channel_gumbel_2_steps
from gesture.utils import read_good_sids,read_sids

fs=1000
channel_num_selected=10
sids=read_sids()
good_sids=read_good_sids()
############### testing accuracy using gumbel/stg/manual/random/all selected channels #############

## get testing result using gumbel method for good subject
def acc_gumebl(sids,channel_num_selected):
    test_acc_gumbel=[]
    for i,sid in enumerate(sids): # or all sids: sids/good_sids
        test_acc_gumbel.append([])
        selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/' + str(channel_num_selected) + 'channels/'
        scores_all = np.load(selection_result + 'score_all.npy', allow_pickle=True).item()  # (train acc, val acc)
        epoch_score = scores_all['epoch_score']
        epoch_score_no_selection = scores_all['epoch_score_no_selection']
        test_acc_gumbel[i] = scores_all['test_acc']
    return test_acc_gumbel

test_acc_gumbel=acc_gumebl(good_sids,channel_num_selected)
## get testing result using channels selected by the stg method (restrict to 10 channels) for good subject #####
test_acc_stg=[]
for i,sid in enumerate(good_sids): # or all sids: sids
    test_acc_stg.append([])
    selection_result = result_dir + 'deepLearning/retrain/stg/10channels/' + 'P' + str(sid) + '/'
    scores_all = np.load(selection_result + 'training_result_500.npy', allow_pickle=True).item()  # (train acc, val acc)
    test_acc_stg[i] = scores_all['test_acc']

## get testing result using stg method (all channels) for good subject #####
test_acc_stg_all=[]
for i,sid in enumerate(good_sids): # or all sids: sids
    test_acc_stg_all.append([])
    selection_result = result_dir + 'deepLearning/retrain/stg/all_selected/' + 'P' + str(sid) + '/'
    scores_all = np.load(selection_result + 'training_result_500.npy', allow_pickle=True).item()  # (train acc, val acc)
    test_acc_stg_all[i] = scores_all['test_acc']


## get testing result using manual selection for all subject
dl_result=result_dir+'selection/manual/P'
test_acc_mannal=[]
window_size=500
for i, sid in enumerate(good_sids):
    test_acc_mannal.append([])
    sid_result=dl_result+str(sid)+'/'
    result_file=dl_result+str(sid)+'/'+'training_result_'+str(window_size)+'.npy'
    result=np.load(result_file,allow_pickle=True).item()
    #train_losses=result['train_losses']
    #train_accs=result['train_accs']
    #val_accs=result['val_accs']
    test_acc_mannal[i]=result['test_acc']

## get testing acc using random channels
test_acc_random=[]
for i,sid in enumerate(good_sids):
    test_acc_random.append([])
    result_file = result_dir + 'selection/random/' + 'P' + str(sid) + '/'+ 'training_result_' + str(window_size) + '.npy'
    result = np.load(result_file, allow_pickle=True).item()  # (train acc, val acc)
    test_acc_random[i] = result['test_acc']

## get testing acc using all channels
dl_result=result_dir+'deepLearning_bak/'
test_acc_all=[]
for i, sid in enumerate(good_sids):
    test_acc_all.append([])
    sid_result=dl_result+str(sid)+'/'
    result_file=dl_result+str(sid)+'/'+'training_result_deepnet'+'.npy'
    result=np.load(result_file,allow_pickle=True).item()
    #train_losses=result['train_losses']
    #train_accs=result['train_accs']
    #val_accs=result['val_accs']
    test_acc_all[i]=result['test_acc']

mean_stg_all = np.mean(test_acc_stg_all) # 0.670+-0.042
mean_gumbel = np.mean(test_acc_gumbel)  # 0.65+-0.035
mean_manual = sum(test_acc_mannal) / len(test_acc_mannal)  # 0.602+-0.031
mean_stg_10 = np.mean(test_acc_stg)  # 0.5958+-0.045
mean_all = np.mean(test_acc_all)  # 0.59+-0.036
mean_random = sum(test_acc_random) / len(test_acc_random)  # 0.36+-0.015
np.var(test_acc_random)

#### line plot all method (not very scientific)
fig, ax=plt.subplots()
ax.plot(test_acc_all,color=colors[0])
ax.plot(test_acc_gumbel,color=colors[1])
ax.plot(test_acc_stg,color=colors[2])
ax.plot(test_acc_stg_all,color=colors[3])
ax.plot(test_acc_mannal,color=colors[4])
ax.plot(test_acc_random,color=colors[5])
ax.axhline(y=mean_all, color=colors[0], linestyle='--')
ax.axhline(y=mean_gumbel, color=colors[1], linestyle='--')
ax.axhline(y=mean_stg_10+0.01, color=colors[2], linestyle='--')
ax.axhline(y=mean_stg_all, color=colors[3], linestyle='--')
ax.axhline(y=mean_manual, color=colors[4], linestyle='--')
ax.axhline(y=mean_random, color=colors[5], linestyle='--')
ax.legend(['all','gumbel','stg_10','stg_all','manual','random'],loc="lower left",bbox_to_anchor=(0.5,0.7,0.1,0.1),fontsize='small')

ax.set_xticks(range(len(sids)))
ax.set_xticklabels(good_sids,rotation = 45, position=(0,0),fontsize='x-small')
filename = result_dir + 'selection/compare_all_line.pdf'
fig.savefig(filename)

#### box plot all method
ax.boxplot([test_acc_stg_all,test_acc_gumbel,test_acc_mannal,test_acc_stg,test_acc_all,test_acc_random],meanline=True,showmeans=True)
filename = result_dir + 'selection/compare_all_box.pdf'
fig.savefig(filename)


import scipy
U1, p = scipy.stats.mannwhitneyu(test_acc_gumbel, test_acc_mannal, method="exact") # p=0.8544656868540574
scipy.stats.ttest_rel(test_acc_gumbel, test_acc_mannal, axis=0, nan_policy='propagate', alternative='two-sided')

###### test the reproductivity of gumbel selection  ############
selected_channels_repeat=[]
test_acc_repeat=[]
channel_num_selected = 10
sid=10
for i in range(39):
    selected_channels_repeat.append([])
    test_acc_repeat.append([])
    selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/' + str(channel_num_selected) + 'channels_'+str(i)+'/'
    score_all = np.load(selection_result + 'score_all.npy', allow_pickle=True).item()  # (train acc, val acc)
    h = np.load(selection_result + 'HH.npy')
    s = np.load(selection_result + 'SS.npy')  # selection
    z = np.load(selection_result + 'ZZ.npy')  # probability
    h = np.squeeze(h)
    z = np.squeeze(z)
    test_acc_repeat[i] = score_all['test_acc'].numpy()
    selected_channels = np.argmax(z[-1, :, :], axis=0)
    selected_channels_repeat[i] = list(set(selected_channels))

channel_num=208
selected_channels_repeat2=[ii for i in selected_channels_repeat for ii in i ]
selected_unique=list(set(selected_channels_repeat2))
selected_unique_count=[]
for i, c in enumerate(range(channel_num)):
    selected_unique_count.append([])
    selected_unique_count[i]=sum([c==i for i in selected_channels_repeat2])
fig, ax=plt.subplots()
ax.bar(range(channel_num),selected_unique_count,width=0.3)
filename=result_dir + 'selection/gumbel/all/10channels/repet_selection_counts.pdf'
fig.savefig(filename)

##### Gumbel decoding accuracy using different channels, tested on sid10
fig, ax=plt.subplots()
acc_vs_channelN=[]
max_channelN=15
sid=10
for i in range(1,max_channelN+1):
    acc_vs_channelN.append([])
    selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/' + str(i) + 'channels'+'/'
    score_all = np.load(selection_result + 'score_all.npy', allow_pickle=True).item()  # (train acc, val acc)
    acc_vs_channelN[i-1]=score_all['test_acc']

ax.plot(list(range(1,max_channelN+1)),acc_vs_channelN)
ax.set_xticks(list(range(1,max_channelN+1)))
ax.set_xlabel('Channel Number', fontsize=10, labelpad=5)
ax.set_ylabel('Decoding Accuracy', fontsize=10, labelpad=5)
filename=result_dir + 'selection/gumbel/all/acc_vs_channelN_sid10.pdf'
fig.savefig(filename)

#### percentage of gumbel selected channel in each ROI #####
import scipy.io
mat=scipy.io.loadmat(mydrive_dir+'matlab/gesture/result/selection/ana_perct_detail.mat')
ana_perct_detail_tmp=mat['ana_perct_detail'].item()
rois_tmp=mat['rois']
ana_selected=[]
ana_all=[]
ana_perct_detail=[]
rois=[]
for i, roi in enumerate(rois_tmp[0]):
    ana_perct_detail.append([])
    ana_selected.append([])
    ana_all.append([])
    rois.append([])
    ana_selected[i]=ana_perct_detail_tmp[i][0,0]
    ana_all[i]=ana_perct_detail_tmp[i][0,1]
    ana_perct_detail[i]=[ana_perct_detail_tmp[i][0,0], ana_perct_detail_tmp[i][0,1]]
    if len(roi)>0:
        rois[i]=roi[0]
    else:
        rois[i] = 'blank'
# exclude the 'nown'(unkonwn)
exclude=rois.index('nown')
rois2,ana_all2,ana_selected2 =[],[],[]
for i in range(len(rois)):
    if i != exclude:
        rois2.append(rois[i])
        ana_all2.append(ana_all[i])
        ana_selected2.append(ana_selected[i])
non_selected=[ana_all2[i]-ana_selected2[i] for i in range(len(ana_selected2))]
## rois2: ['Hippocampus', 'Inf_Lat_Vent', 'UnsegmentedWhiteMatter', 'White_Matter', 'central', 'insula',
# 'lateraloccipital', 'lingual', 'paracentral', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
# 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal']
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
width=0.35
ax.bar(range(len(rois2)), non_selected, width, label='All Channels')
ax.bar(range(len(rois2)), ana_selected2, width, bottom=non_selected,label='Selected Channels')
ax.set_xticks(range(len(rois)))
filename=result_dir + 'selection/gumbel/all/10channels/roi_percentage.pdf'
fig.savefig(filename)

##############  bar plot of the channel number selected by STG, Gumbel and both
stg_count=[5,29,5,11,13,6,49,29,22,7,3,18]
gumbel_count=[3,2,0,0,0,6,0,1,3,4,5,1]
common=[8,8,7,10,9,4,10,8,7,4,5,8]
width = 0.35

fig, ax = plt.subplots()
ax.bar(np.arange(1,len(good_sids)+1), stg_count, width, label='STG')
ax.bar(np.arange(1,len(good_sids)+1), common, width, bottom=stg_count, label='Common')
ax.bar(np.arange(1,len(good_sids)+1), gumbel_count, width, bottom=[common[i]+stg_count[i] for i in range(len(common))], label='Gumbel')
ax.legend()
ax.set_xticks(np.arange(1,len(good_sids)+1))
ax.set_xticklabels(np.arange(1,len(good_sids)+1))
ax.set_ylim([0,70])
filename=result_dir + 'selection/compare_channel_count.pdf'
fig.savefig(filename)


