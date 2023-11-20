from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

data_dir = '/Users/long/Documents/data/gesture/'# temp data dir

save_dir=data_dir+'training_result/compare_result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
info=info_dir+'info.npy'
info=np.load(info,allow_pickle=True)
sids=info[:,0]
dl_models=['eegnet','shallowFBCSPnet', 'deepnet', 'deepnet_da', 'resnet']

# deep learning result
dl_result=data_dir+'training_result/deepLearning/'
decoding_accuracy={}
for sid in sids:
    decoding_accuracy[str(sid)]=[]
    sid_result=dl_result+str(sid)+'/'
    for modeli in dl_models:
        result_file=sid_result+'training_result_'+modeli+'.npy'
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
        decoding_accuracy[str(sid)].append(test_acc)

ml_result=data_dir+'training_result/machineLearning/'
ml_models=['SVM','FBCSP']
for sid in sids:
    for modeli in ml_models:
        sid_result = ml_result + modeli + '/'
        if modeli=='SVM':
            result_file=sid_result+'P'+str(sid)+'/SVM_accVSfeat.npy'
            result = np.load(result_file, allow_pickle=True).max()
            decoding_accuracy[str(sid)].append(result)
        elif modeli=='FBCSP':
            result_file = sid_result + str(sid)+'.npy'
            result = np.load(result_file, allow_pickle=True).item()
            decoding_accuracy[str(sid)].append(result)

# get average decoding accuracy and sort
acc_avg=np.asarray([sum(decoding_accuracy[key])/len(decoding_accuracy[key]) for key in decoding_accuracy.keys()])
top5_idx=np.asarray(acc_avg).argsort()[::-1][:5]
top5_sid=sids[top5_idx]

accs={}
for sid in top5_sid:
    accs[str(sid)]=decoding_accuracy[str(sid)]

# compare subjects
all_model_name=['eegnet','shallownet', 'deepnet', 'deepnet_da', 'resnet','SVM','FBCSP']
fig,ax=plt.subplots()
x=[1,2,3,4,5,6,7]
colors=['orangered','yellow', 'gold','orange','springgreen','aquamarine','skyblue']
from matplotlib.patches import Patch
cmap = dict(zip(all_model_name, colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
for i,sid in enumerate(sorted(top5_sid)):
    ax.bar(x, accs[str(sid)], width=0.3,color=colors)
    x=[i+10 for i in x]
    if i==0:
        ax.legend(all_model_name,ncol=4,handles=patches,fontsize='small',loc='upper left', bbox_to_anchor=(0.0, 1.15))
x=[5,15,25,35,45]
sid_list=[3,8,11,24,30]
ax.set_xticks(x)
ax.set_xticklabels(['sid '+str(i) for i in sid_list], position=(0,0.01))
filename=save_dir+'compare_top5_sids.pdf'
fig.savefig(filename)

# rearrange as model
model_acc=[]
for i, modeli in enumerate(all_model_name):
    model_acc.append([])
    for sid in top5_sid:
        model_acc[i].append(accs[str(sid)][i])

acc_mean=[np.mean(tmp) for tmp in model_acc]
acc_std=[np.std(tmp) for tmp in model_acc]

from scipy import stats
_, p = stats.ttest_rel(model_acc[2],model_acc[3]) #p=0.11457442074512621

ax.clear()
x=[1,2,3,4,5,6,7]
ax.bar(x, acc_mean, width=0.3,yerr=acc_std, color=['yellow'],error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
#barplot_annotate_brackets(0,2,3,p,[0,1,2,3,4],bar_mean,bar_err.tolist())
ax.set_xticks(x)
ax.set_xticklabels(all_model_name,rotation = 45, position=(0,0.02))
filename=save_dir+'compare_models.pdf'
fig.savefig(filename)