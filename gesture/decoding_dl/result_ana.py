import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *
from gesture.utils import get_decoding_acc
from pre_all import *
from common_plot import barplot_annotate_brackets

#data_dir = '/Users/long/Documents/data/gesture/'# temp data dir

save_dir=result_dir+'compare_result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
info=info_dir+'info.npy'
info=np.load(info,allow_pickle=True)
sids=info[:,0]
#sids=good_sids
dl_models=['eegnet','FBCSP', 'shallowFBCSPnet', 'deepnet', 'deepnet_da', 'resnet']


################# select good_sids using deepnet #############
method='vanilla' # using deepnet without data augmentation
acc_dict= get_decoding_acc(sids,method)
keys=[str(sid) for sid in sids]
values=np.asarray([acc_dict[str(sid)] for sid in sids])

good_index=np.where(values>0.4)[0]
good_sids=sids[good_index] # [ 2,  3,  4, 10, 13, 17, 18, 29, 32, 41]

fig, ax=plt.subplots(2,1)
ax[0].plot(values)
ax[0].set_xticks(range(len(sids)))
ax[0].set_xticklabels(sids)
ax[1].plot(values[good_index])
ax[1].set_xticks(range(len(good_index)))
ax[1].set_xticklabels(good_sids)

# save good_sid to text file
outfile=meta_dir+'good_sids.txt'
if os.path.exists(outfile):
    os.remove(outfile)
with open(outfile, 'w') as f:
    for item in good_sids:
        f.write("%d\n" % item)

#### decoding accuracy using NI DA ####
method='NI' # using deepnet without data augmentation
acc_dict= get_decoding_acc(sids,method)
keys=[str(sid) for sid in sids]
acc_NI=np.asarray([acc_dict[str(sid)] for sid in sids])
ax[0].plot(acc_NI)

####### plot deep and machine learning decoding accuracy for all subjects ##########
#TODO: rewrite this to use get_decoding_acc(sids, method);
dl_result=result_dir+'deepLearning_bak/'
ml_result=result_dir+'machineLearning/FBCSP/'
da_dl_ml={}
for sid in sids:
    da_dl_ml[str(sid)]=[]
    sid_result=dl_result+str(sid)+'/'
    for modeli in dl_models:
        if modeli=='FBCSP':
            # append machine learning result
            sid_result_file = ml_result + str(sid) + '.npy'
            test_acc = np.load(sid_result_file, allow_pickle=True).item()
            da_dl_ml[str(sid)].append(test_acc)
        else:
            result_file=sid_result+'training_result_'+modeli+'.npy'
            result=np.load(result_file,allow_pickle=True).item()

            train_losses=result['train_losses']
            train_accs=result['train_accs']
            val_accs=result['val_accs']
            test_acc=result['test_acc']
            da_dl_ml[str(sid)].append(test_acc)

# only keep the good_sids
da_dl_ml_good_sids={}
for sid in good_sids:
    da_dl_ml_good_sids[str(sid)]=da_dl_ml[str(sid)]

# convert dic to numpy array: good_sid
model_number=len(da_dl_ml_good_sids['10'])
sid_number=len(da_dl_ml_good_sids)
da_result=np.zeros((model_number,sid_number))
for i,k in enumerate(da_dl_ml_good_sids):
    da_result[:,i]=da_dl_ml_good_sids[str(k)]
# convert dic to numpy array: all sids
sid_number=len(da_dl_ml)
da_result_all=np.zeros((model_number,sid_number))
for i,k in enumerate(da_dl_ml):
    da_result_all[:,i]=da_dl_ml[str(k)]

fig,ax=plt.subplots()
colors=['orangered','yellow', 'gold','orange','springgreen','aquamarine']#,'skyblue']
all_models_name=['EEGNet','FBCSP','shallownet', 'deepnet', 'STSCNet', 'ResNet']
for i in range(model_number): ax.plot(da_result_all[i,:],c=colors[i])
ax.legend(all_models_name,loc="lower left",bbox_to_anchor=(0.5,0.6,0.1,0.1),fontsize='small')
ax.set_xticks(range(len(sids))) #指定要标记的坐标
ax.set_xticklabels([*range(1, 31, 1)],rotation = 45, position=(0,0),fontsize='xx-small')
filename = result_dir + 'compare_result/acc_all_dl_ml.pdf'
fig.savefig(filename)

############# average decoding accuracy of all models, and bar plot it ###############
ax.clear()
da_result_model=np.average(da_result,axis=1) #0.39, 0.63 , 0.64, 0.62, 0.68,0.4
x=[1,2,3,4,5,6] # 6 models

# bar plot
from matplotlib.patches import Patch
cmap = dict(zip(all_models_name, colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
ax.bar(x, da_result_model, width=0.3,color=colors)
ax.set_ylim([0, 1])
ax.legend(all_models_name,ncol=3,handles=patches,fontsize='small',loc='upper left', bbox_to_anchor=(0.0, 1.15))
filename = result_dir + 'compare_result/mean_acc_all_dl_ml.pdf'
fig.savefig(filename)

# dot plot with error bar
ax.clear()
ax.scatter(x, da_result_model)
min_max=np.zeros((2,6)) # 6 models
for modeli in range(6):
    min_max[0,modeli]=da_result_model[modeli]-min(da_result[modeli,:])
    min_max[1, modeli] = max(da_result[modeli, :])-da_result_model[modeli]
ax.errorbar(x, da_result_model, yerr=min_max, fmt="o")
filename = result_dir + 'compare_result/mean_acc_all_dl_ml_dot_min_max.pdf'
fig.savefig(filename)

# violin plot
## combine these different collections into a list
ax.clear()
ax.violinplot(da_result.transpose(),showmeans=True)
ax.set_xticks([1,2,3,4,5,6]) #指定要标记的坐标
ax.set_xticklabels(all_models_name,rotation = 0, position=(0,0),fontsize=10)
ax.set_ylabel('Decoding Accuracy', fontsize=10, labelpad=5)
filename = result_dir + 'compare_result/mean_acc_all_dl_ml_violin.pdf'
fig.savefig(filename)

############# compare training process and result between full and selected channels using deepnet model ############

## read data into dictionaries
train_accs={}
val_accs={}
test_acc={}
train_accs_gumble={}
val_accs_gumble={}
test_acc_gumble={}
for sid in good_sids:
    key='sid'+str(sid)
    sid_result = dl_result + str(sid) + '/'
    result_file = sid_result + 'training_result_deepnet*' + '.npy'
    result_file = os.path.normpath(glob.glob(result_file)[0])
    result = np.load(result_file, allow_pickle=True).item()
    train_accs[key] = result['train_accs']
    val_accs[key] = result['val_accs']
    test_acc[key] = result['test_acc']

    sid_result = dl_result + 'selected_channel_gumble/' + str(sid) + '/'
    result_file = sid_result + 'training_result_deepnet*' + '.npy'
    result_file = os.path.normpath(glob.glob(result_file)[0])
    result_gumble_selection = np.load(result_file, allow_pickle=True).item()
    train_accs_gumble[key] = result_gumble_selection['train_accs']
    val_accs_gumble[key] = result_gumble_selection['val_accs']
    test_acc_gumble[key] = result_gumble_selection['test_acc']

## check individual sid (good_sids: [2, 3, 4, 10, 13, 17, 18, 25, 29, 32, 34, 41])
fig, ax=plt.subplots()
for sid in good_sids:
    #sid=3
    ax.clear()
    key='sid'+str(sid)
    train_acci=train_accs[key]
    val_acci=val_accs[key]
    ax.plot(train_acci,'b')
    ax.plot(val_acci,'r')
    epoch_number=len(train_acci)
    train_acci_gumble=train_accs_gumble[key]
    val_acci_gumble=val_accs_gumble[key]
    epoch_number=len(train_acci_gumble)
    ax.plot(train_acci_gumble[:epoch_number],'b-.')
    ax.plot(val_acci_gumble[:epoch_number],'r-.')
    ax.legend(['train_acc','val_acc','train_acc_gumble','val_acc_gumble'],loc="lower left",bbox_to_anchor=(0.6,0.5,0.1,0.1),fontsize='small')
    filename = dl_result + 'selected_channel_gumble/fullVSgumble_sid' + str(sid)
    fig.savefig(filename)


# lenght=12
test_acc_list=[test_acc['sid'+str(i)] for i in good_sids ]
test_acc_gumble_list=[test_acc_gumble['sid'+str(i)] for i in good_sids ]
ax.plot(list(range(12)),test_acc_list,'ro')
ax.plot(list(range(12)),test_acc_gumble_list,'bo')
ax.legend(['all channel', 'selected'], loc="lower left",bbox_to_anchor=(0.6, 0.85, 0.1, 0.1), fontsize='small')
ax.set_xticks(list(range(12))) #指定要标记的坐标
ax.set_xticklabels(['sid'+str(i) for i in good_sids],rotation = 45,position=(0,0))
filename = dl_result + 'selected_channel_gumble/acc_fullVSgumble'
fig.savefig(filename)

