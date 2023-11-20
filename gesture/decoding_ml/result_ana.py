from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

sids=good_sids

ml_result=result_dir+'machineLearning/'
ml_models=['SVM','FBCSP']
da_ml={}
for sid in sids:
    da_ml[str(sid)] = []
    for modeli in ml_models:
        sid_result = ml_result + modeli + '/'
        if modeli=='SVM':
            result_file=sid_result+'P'+str(sid)+'/SVM_accVSfeat.npy'
            result = np.load(result_file, allow_pickle=True).max()
            da_ml[str(sid)].append(result)
        elif modeli=='FBCSP':
            result_file = sid_result + str(sid)+'.npy'
            result = np.load(result_file, allow_pickle=True).item()
            da_ml[str(sid)].append(result)
svm_acc=[da_ml[str(i)][0] for i in sids]
fig,ax=plt.subplots()
ax.plot(svm_acc)
ax.set_xticks(range(len(sids)))
ax.set_xticklabels(sids)
top_sids=np.where(np.asarray(svm_acc)>0.4)[0].tolist()
good_sids=[sids[i] for i in top_sids] # [3, 4, 10, 13, 29, 41]





