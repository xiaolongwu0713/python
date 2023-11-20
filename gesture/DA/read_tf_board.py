import matplotlib.pyplot as plt
import numpy as np
import scipy

from gesture.config import *
from gesture.utils import read_good_sids
from tbparse import SummaryReader
from pre_all import *
#### model selection according to the training process ####
fig,ax=plt.subplots()
event_dir='H:/Long/data/gesture/DA/TTSCGAN/3/2022_11_18_19_41_49/Log/'
#
file=event_dir+'monitor_r_acc_adv/events.out.tfevents.1668771710.workstation.45308.6'
reader = SummaryReader(file) # long format
df = reader.scalars
r_acc_adv=df.iloc[:,2]
#
file=event_dir+'monitor_f_acc_adv/events.out.tfevents.1668771710.workstation.45308.7'
reader = SummaryReader(file) # long format
df = reader.scalars
f_acc_adv=df.iloc[:,2]
#
file=event_dir+'monitor_r_acc_cls/events.out.tfevents.1668771710.workstation.45308.8'
reader = SummaryReader(file) # long format
df = reader.scalars
r_acc_cls=df.iloc[:,2]
#
file=event_dir+'monitor_f_acc_cls/events.out.tfevents.1668771710.workstation.45308.9'
reader = SummaryReader(file) # long format
df = reader.scalars
f_acc_cls=df.iloc[:,2]

all_events=[r_acc_adv,f_acc_adv,r_acc_cls,f_acc_cls]
ax.clear()
for i in range(len(all_events)):
    # original event data
    #ax.plot(all_events[i],linewidth=0.2,alpha=0.1,color=tab_colors_codes[i])
    # smoothing the data
    ax.plot(smooth(all_events[i].to_numpy(),100),linewidth=0.5,alpha=1,color=tab_colors_codes[i])
ax.legend(['r_acc_adv','f_acc_adv','r_acc_cls','f_acc_cls'],fontsize=10)
filename=result_dir+'DA/plots/training_process_standard_loss.pdf'
fig.savefig(filename)

## d_loss reach to 0 using WGANGP method ###
event_dir='H:/Long/data/gesture/DA/CWGANGP/10/2022_11_23_11_42_26/events.out.tfevents.1669174950.workstation.2236.0'
reader = SummaryReader(event_dir) # long format
df = reader.scalars
d_loss=df[df['tag']=='d_loss']['value'].to_numpy()
ax.clear()
ax.plot(smooth(d_loss,100),linewidth=0.7)
filename=result_dir+'DA/plots/training_process_wgangp_d_loss.pdf'
fig.savefig(filename)

## d_loss using TTSCGAN method ###
event_dir='H:/Long/data/gesture/DA/TTSCGAN/10/2022_12_03_21_39_27/Log/events.out.tfevents.1670074770.workstation.57948.0'
reader = SummaryReader(event_dir) # long format
df = reader.scalars
d_loss=df[df['tag']=='d_loss']['value'].to_numpy()
ax.clear()
ax.plot(smooth(d_loss,100)[:-100]+1,linewidth=0.7)
filename=result_dir+'DA/plots/training_process_ttscgan_d_loss.pdf'
fig.savefig(filename)

## Item: read images
fig,ax=plt.subplots()
event_dir='H:/Long/data/gesture/DA/TTSCGAN/10/2022_12_01_10_59_14/Log/events.out.tfevents.1669863557.workstation.33136.0'
reader = SummaryReader(event_dir)
imgs=reader.images
image=imgs.iloc[335,:]
a=image.value
ax.imshow(a)
ax.clear()
filename=result_dir+'DA/plots/bad_example.pdf'
fig.savefig(filename)







