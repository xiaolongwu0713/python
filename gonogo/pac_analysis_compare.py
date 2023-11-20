import hdf5storage
import matplotlib.pyplot as plt
import scipy.io
import os
import math
from gonogo.config import *
import numpy as np

save_dir=result_dir+'pac_plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sid=5
indicator='result_sig' # 'result_sig'/'result'
filename=matlab_result_dir+str(sid)+'_epoch11.mat'
mat=scipy.io.loadmat(filename)
raw1=mat[indicator].item() # raw is a np structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
del mat
filename=matlab_result_dir+str(sid)+'_epoch12.mat'
mat=scipy.io.loadmat(filename)
raw2=mat[indicator].item() # raw is a np structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
del mat

if indicator=='result_sig':
    raw1=[np.negative(np.log(tuplei)) for tuplei in raw1]
    raw2 = [np.negative(np.log(tuplei)) for tuplei in raw2]

chnNum=len(raw1) # (14, 91)

channel1 = range(chnNum)  # [i-1 for i in [1,2]]
channel2 = channel1  # [i for i in range(63)]
freq4phase = range(2,15+1,1)
freq4power = range(20,200+2,2) #20:10:100;

xtickNum=6
pointsN=math.ceil(len(freq4power)/6)
xtick=[]
for i in range(xtickNum):
    xtick.append(i*pointsN)
xtick.append(len(freq4power))

xtickLabels=[]
for i in range(xtickNum):
    xtickLabels.append(freq4power[i*pointsN])
xtickLabels.append(freq4power[-1])

ytick=list(range(raw1[0].shape[0]))
ytickLabel=list(range(2,15+1,1))
'''
# plot 20 pac/page
cue_on_idex=5
cue_off_idex=35
vmin=-4
vmax=4
plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
#plt.rc('y', pad=0)
fig, ax = plt.subplots(10, 5, figsize=(6, 3), sharex=True)
for c1 in channel1:
    # i-th image
    if c1 != 0:
        ax=fig.subplots(10, 5, sharex=True)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    for c2 in channel2:
        #j-th row
        for k,f in enumerate(freq4phase):
            #f-th column
            com=str('c'+str(c1+1)+'_'+str(c2+1))
            datai=raw[com][0][0]
            axi=ax[c2,k]
            im0 = axi.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axi.axvline(x=cue_on_idex, linewidth=1, color='r', linestyle='--')
            axi.axvline(x=cue_off_idex, linewidth=1, color='r', linestyle='--')
            axi.yaxis.set_tick_params(labelsize=4)  # ticklable smaller小坐标轴
            axi.tick_params(axis='y', which='major', pad=0.0)  # space between ticklabel and yaxis，坐标轴紧凑
             # orleft=0.1,bottom=0.1,right=0.9,top=0.9,子图靠近。
    fig.suptitle("Channel " + str(c1)+ ". Low frequency(column):2,4,6,8,10 Hz; High f(row): 20:20:100;",fontsize=8)
    save_file=save_dir+com+'.pdf'
    fig.savefig(save_file)
    fig.clear()
'''

# ad-hoc
#freq4phase = 2:1:15; % Hz
#freq4power = 20:2:150;
cue_on_idex=5
cue_off_idex=35
if indicator=='result_sig':
    vmin=-10
    vmax=10
else:
    vmin = -10
    vmax = 10
plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
#plt.rc('y', pad=0)

ppp=4 # plots per figure
figs=math.ceil(chnNum / ppp) # figure number

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
for i, f in enumerate(range(figs)):
    for j in range(ppp):
        chni=i*ppp+j
        if chni<chnNum:
            inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[j], hspace=0.01)
            ax0 = plt.Subplot(fig, inner[0])
            datai=raw1[chni] # (14, 66)
            im0 = ax0.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax0.set_aspect('auto')
            ax0.set_yticks(ytick)
            ax0.set_yticklabels(ytickLabel, position=(0, -0.04))
            fig.add_subplot(ax0)
            bottom_side = ax0.spines["bottom"]
            bottom_side.set_visible(False)
            ax0.axes.xaxis.set_visible(False)

            ax1 = plt.Subplot(fig, inner[1])
            ax1.sharex(ax0)
            top_side = ax1.spines["top"]
            top_side.set_visible(False)
            datai = raw2[chni]  # (14, 66)
            im0 = ax1.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax1.set_aspect('auto')
            ax1.set_xticks(xtick)
            ax1.set_xticklabels(xtickLabels,position = (0, -0.04))
            ax1.set_yticks(ytick)
            ax1.set_yticklabels(ytickLabel, position=(0, -0.04))

            fig.add_subplot(ax1)
    filename = save_dir + 'PCA_compare_' + indicator+str(sid) + '_' + str(chni) + '.pdf'
    fig.savefig(filename, dpi=400)
    fig.clear()







