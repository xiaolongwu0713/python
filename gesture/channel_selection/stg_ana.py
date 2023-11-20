import hdf5storage
import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from gesture.channel_selection.utils import get_good_sids, get_good_sids_gumbel,\
    get_selected_channel_gumbel, get_selected_channel_stg

####### compare decoding accuracy of all good subjects using different lambda values ############
lam=[0.02, 0.1, 0.2, 0.6, 1.0, 3.0]
good_sids=get_good_sids()
acc=np.zeros((len(lam),len(good_sids)))
for l, lami in enumerate(lam):
    for s, sid in enumerate(good_sids):
        result_file=result_dir+'selection/stg/lambda/'+str(lami)+'/P'+str(sid)+'/result'+str(sid)+'.npy'
        result = np.load(result_file, allow_pickle=True).item()
        acc[l,s]=result['test_acc']
fig,ax=plt.subplots()

# line plot
means=acc.mean(axis=1) # [0.58819444, 0.59212963, 0.62662037, 0.48240741, 0.49467593, 0.41527778]
for i in range(acc.shape[0]):
    ax.plot(acc[i], color=colors[i])
ax.legend([str(i) for i in lam],loc="lower left",bbox_to_anchor=(0.5,0.6,0.1,0.1),fontsize='small')
for i in range(acc.shape[0]):
    ax.axhline(y=means[i], color=colors[i], linestyle='--')

ax.set_xticks(range(acc.shape[1]))
ax.set_xticklabels(range(1,13),rotation = 0, position=(0,0),fontsize='x-small')
ax.set_ylabel('Decoding Accuracy', fontsize=16, labelpad=5)
filename=result_dir+'selection/stg/compare_lambda.pdf'
fig.savefig(filename)

# error box plot
acc_means=acc.mean(axis=1)
green_circle = dict(markerfacecolor='g', marker='o')
ax.boxplot(acc.transpose(),flierprops=green_circle, usermedians=acc_means)
ax.set_ylim([0, 1.3])
#acc_medians=np.median(acc,axis=1)
for i in range(6):
    ax.annotate(str(round(acc_means[i], 2)),(i+1-0.2,acc_means[i]+0.02))
filename=result_dir+'selection/stg/compare_lambda_box.pdf'
fig.savefig(filename)

from scipy.stats import mannwhitneyu
U1, p = mannwhitneyu(acc[-1,:], acc[5,:], method="exact") # p= 0.1, 0.05, 0.01, 0.6, 0.4,,1.0


##########  check the decoding accuracy, and selected channels ############
test_acc=[]
prob_vs_epoch=[]
selected_channels=[]
selected_number=[]
for i, sid in enumerate(good_sids):
    selected_channels.append([])
    test_acc.append([])
    prob_vs_epoch.append([])
    selected_channels[i], test_acc[i], prob_vs_epoch[i]=get_selected_channel_stg(sid, 9999)
    selected_number.append(len(selected_channels[i]))
# channel prob for each subject
for i in range(6):
    plt.plot(all_probs[i])

fig,ax=plt.subplots()
# pro vs epoch for each sid
ax.imshow(prob_vs_epoch[2])
prob=prob_vs_epoch[2][-1,:]


#### test the entropy ####
## good separation between two groups and small variance winthin.
from scipy.stats import entropy
channel_number=150
probs=1/channel_number
a=probs/2
b=probs+a

dis=(b-a) * np.random.random_sample((channel_number,)) +a # sample from range: [0.0, 1.0)
dis=dis/dis.sum()
entropy(dis, base=2)

dis2=dis/2
dis2[np.random.randint(1,channel_number,size=20)]=1
dis2=dis2/dis2.sum()
entropy(dis2, base=2)

plt.plot(dis2)

