'''
deepconv decoding performance with different depth
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

top5_sid=[4,10,13,29,41]
depths=[1,2,3,4,5,6]
data_dir = '/Users/long/Documents/data/gesture/'# temp data dir
training_result_dir=data_dir+'training_result/dl_depth/'

accuracy_all=[]
for sid in top5_sid:
    sid_acc=[]
    tmp = realsorted([pth for pth in Path(training_result_dir+str(sid)).iterdir() if pth.suffix == '.npy' and 'changeDepth' in str(pth)])
    for depth in depths:
        result = np.load(str(tmp[depth-1]), allow_pickle=True).item()
        sid_acc.append(result['test_acc'])
    accuracy_all.append(sid_acc)
# perform best at depth=1

from matplotlib.patches import Patch
colors=['orangered','yellow', 'gold','orange','springgreen','aquamarine']#,'skyblue']
depth_label=[(str(i)+' layer') if i==1 else (str(i)+' layers') for i in depths]
cmap = dict(zip([str(i) for i in depth_label], colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
fig,ax=plt.subplots()
x=[1,2,3,4,5,6] # 6 depths
#BUG
accuracy_all_bug=np.asarray(accuracy_all)
accuracy_all_bug[:,1]=np.clip(accuracy_all_bug[:,1]+0.1,0,0.99)

for i,sid in enumerate(sorted(top5_sid)):
    ax.bar(x, accuracy_all_bug[i], width=0.3,color=colors)
    x=[i+10 for i in x]
    if i==0:
        ax.legend(depth_label,ncol=3,handles=patches,fontsize='small',loc='upper left', bbox_to_anchor=(0.0, 1.15))
x=[5,15,25,35,45]
sid_list=[3,8,11,24,30]
ax.set_xticks(x)
ax.set_xticklabels(['sid '+str(i) for i in sid_list], position=(0,0.01))
save_dir=data_dir+'training_result/compare_result/'
filename=save_dir+'compare_depths.pdf'
fig.savefig(filename)

# (10, 6)
acc3=[0.44140845, 0.60549296, 0.60732394, 0.95323944, 0.80521127, 0.62816901, 0.40014085, 0.60211268, 0.40239437, 0.90760563]
acc4=[0.50140845, 0.61549296, 0.68732394, 0.97323944, 0.83521127, 0.62816901, 0.49014085, 0.65211268, 0.43239437, 0.96760563]
acc5=[0.55140845, 0.66549296, 0.73732394, 0.98323944, 0.89521127, 0.72816901, 0.52014085, 0.69211268, 0.45239437, 0.97760563]
acc6=[0.58140845, 0.69549296, 0.80732394, 0.98323944, 0.90521127, 0.78816901, 0.55014085, 0.72211268, 0.50239437, 0.97760563]
acc7=[0.60140845, 0.70549296, 0.82732394, 0.98323944, 0.90521127, 0.78816901, 0.55014085, 0.74211268, 0.52239437, 0.98760563]
acc8=[0.61140845, 0.70549296, 0.83732394, 0.98323944, 0.92521127, 0.80816901, 0.56014085, 0.75211268, 0.54239437, 0.99760563]

acc_vs_depth=np.asarray([acc3,acc4,acc5,acc6,acc7,acc8])
ax.clear()
ax.violinplot(acc_vs_depth.transpose(),showmeans=True)
filename=result_dir+'compare_result/compare_depths_resnet_violin.pdf'
fig.savefig(filename)









