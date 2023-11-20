import hdf5storage
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
data_dir='/Users/long/Documents/data/work/ruijin/gonogo/'
filename=data_dir+'sid6pac_result.mat'
save_dir=data_dir+'pac_plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#mat2=hdf5storage.loadmat(filename)

mat=scipy.io.loadmat(filename)
raw=mat['result'] # raw is a np structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
del mat

channel1 = range(63)  # [i-1 for i in [1,2]]
channel2 = channel1  # [i for i in range(63)]
freq4phase = range(2,15+1,1)
freq4power = range(20,150+2,2) #20:10:100;

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

# ad hot
#freq4phase = 2:1:15; % Hz
#freq4power = 20:2:150;
cue_on_idex=5
cue_off_idex=35
vmin=-10
vmax=10
plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
#plt.rc('y', pad=0)
fig, ax = plt.subplots(3, 3, figsize=(6, 3), sharex=True,sharey=True)
figi=0 # the first figi
for i,c1 in enumerate(channel1):
    c2=c1
    i=i-figi*9
    rowi=i//3
    coli=i%3

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    axi=ax[rowi,coli]

    com=str('c'+str(c1+1)+'_'+str(c2+1))
    datai=raw[com][0][0] # (14, 66)

    im0 = axi.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    #axi.axvline(x=cue_on_idex, linewidth=1, color='r', linestyle='--')
    #axi.axvline(x=cue_off_idex, linewidth=1, color='r', linestyle='--')

    xloc = range(0, datai.shape[1])[::5]
    xval = range(20, 150 + 2, 2)[::5]

    yloc=range(datai.shape[0])[::2]
    yval=range(2,16)[::2]

    axi.set_xticks(xloc)
    axi.set_xticklabels(xval)
    axi.set_yticks(yloc)
    axi.set_yticklabels(yval)

    axi.yaxis.set_tick_params(labelsize=4)  # ticklable smaller小坐标轴
    axi.tick_params(axis='y', which='major', pad=0.0)  # space between ticklabel and yaxis，坐标轴紧凑
         # orleft=0.1,bottom=0.1,right=0.9,top=0.9,子图靠近。
    if i>0 and (i+1)%9==0:
        fig.suptitle("Channel " + str(figi*9+1)+ "_"+ str(figi*9+9),fontsize=8)
        save_file=save_dir+str(figi*9+1)+ "_"+ str(figi*9+9)+'.pdf'
        fig.savefig(save_file)
        fig.clear()
        ax = fig.subplots(3, 3, sharex=True,sharey=True)
        figi=figi+1









