'''
resnet decoding performance with different window size
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

top5_sid=[4,10,13,29,41]
winds=[100,200,300,400,500]
data_dir = '/Users/long/Documents/data/gesture/'# temp data dir
training_result_dir=data_dir+'training_result/wind_size/'

accuracy_all=[]
for i,wind in enumerate(winds):
    accuracy_all.append([])
    for sid in top5_sid:
        filename=training_result_dir+str(wind)+'ms/'+str(sid)+'/training_result_resnet.npy'
        result = np.load(filename, allow_pickle=True).item()['test_acc']
        accuracy_all[i].append(result)
# perform best at depth=1
accuracy_all=np.asarray(accuracy_all)# (sid,wind_size)

from matplotlib.patches import Patch
colors=['orangered','yellow', 'gold','orange','springgreen']#,'aquamarine']#,'skyblue']
wind_label=[(str(i)+' ms')  for i in winds]
cmap = dict(zip([str(i) for i in wind_label], colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
fig,ax=plt.subplots()

ax.clear()
x=[1,2,3,4,5] # 5 wind sizes
for i,sid in enumerate(sorted(top5_sid)):
    ax.bar(x, accuracy_all[i], width=0.3,color=colors)
    x=[i+10 for i in x]
    if i==0:
        ax.legend(wind_label,ncol=3,handles=patches,fontsize='small',loc='upper left', bbox_to_anchor=(0.0, 1.15))

x=[3,13,23,33,43]
sid_list=[3,8,11,24,30]
ax.set_xticks(x)
ax.set_xticklabels(['sid '+str(i) for i in sid_list], position=(0,0.01))
save_dir=data_dir+'training_result/compare_result/'
filename=save_dir+'compare_wind_size.pdf'
fig.savefig(filename)



