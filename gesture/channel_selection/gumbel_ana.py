# sid10(208channels): [143, 144, 146, 147, 148, 149, 150, 151, 152, 167]
# sid41(190channels): [33, 163, 37, 166, 165, 19, 20, 26, 30, 63]

import hdf5storage
import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from gesture.channel_selection.utils import get_selected_channel_gumbel_2_steps

sid=10
fs=1000
channel_num_selected=10


## training process of the selection network for one subject
selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/' + str(
    channel_num_selected) + 'channels/'
# result_dir='/Users/long/OneDrive/share/selection/gumbel/3/P10'
scores_all = np.load(selection_result + 'score_all.npy', allow_pickle=True).item()  # (train acc, val acc)
epoch_score=scores_all['epoch_score']
epoch_score_no_selection=scores_all['epoch_score_no_selection']
test_acc=scores_all['test_acc']

h = np.load(selection_result + 'HH.npy')
s = np.load(selection_result + 'SS.npy')  # selection
z = np.load(selection_result + 'ZZ.npy')  # probability
h = np.squeeze(h)
z = np.squeeze(z)
mean_entropy = np.mean(h, axis=1)

fig, ax=plt.subplots()
x1=range(len(epoch_score))
train_acc=[i[0] for i in epoch_score]
eval_acc=[i[1] for i in epoch_score]
ax.plot(x1, train_acc, color='red')
ax.plot(x1, eval_acc, color='blue')
ax.plot(x1, mean_entropy, color='yellow')

train_acc2=[i[0] for i in epoch_score_no_selection]
eval_acc2=[i[1] for i in epoch_score_no_selection]
x2=[i+len(x1)+1 for i in range(len(train_acc2))]
ax.plot(x2, train_acc2, color='orchid')
ax.plot(x2, eval_acc2, color='lightblue')

filename=selection_result+'training_result.pdf'
fig.savefig(filename)

zz=np.mean(z,axis=2)
im=ax.imshow(zz)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)
filename=selection_result+'probability_process.pdf'
fig.savefig(filename)

