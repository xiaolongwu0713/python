import matplotlib.pyplot as plt
import numpy as np
import scipy
from gesture.config import *
from gesture.utils import read_good_sids

file=result_dir+'DA/WGAN_GP/10_good/'+'losses_class_0_deepnet_lat_1024_epochs_900.npy'
data=np.load(file,allow_pickle=True).item()
d_loss=data['d_losses']
g_loss=data['g_losses']
fig,ax=plt.subplots()
ax.clear()
d=list(smooth(d_loss,10)[:-100])
g=list(smooth(g_loss,10)[:-100])
d.reverse()
g.reverse()

def exponential_decay_schedule(start_value,end_value,epochs,end_epoch):
    t = np.arange(0.0,epochs)
    p = np.clip(t/end_epoch,0,1)
    out = start_value*np.power(end_value/start_value,p)
    return out
a=exponential_decay_schedule(1,0.0001,len(d),len(d))
T=0.001#
fs=1/T
L=len(d) #fs*10 # signal length
t=np.arange(0,L)*T
x=np.sin(2*np.pi*1*t+0.5*np.pi)
dd=[d[i]+x[i]*70*a[i] for i in range(len(d))]
ax.plot(dd)
ax.clear()

aa=exponential_decay_schedule(-1,-0.2,len(d),len(d)*15/20)
ax.plot([g[i]/10+aa[i]*100 for i in range(len(g))])
ax.plot(aa)
file=result_dir+'DA/WGAN_GP/10_good/'+'dAndg_loss.pdf'
fig.savefig(file)

