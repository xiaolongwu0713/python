# visualization data generated and the original data with tSNE

import numpy as np
from gesture.DA.metrics.visualization import visualization
from gesture.channel_selection.utils import get_selected_channel_gumbel
from gesture.utils import read_data_split_function, windowed_data
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
sid=10
classi=0
fs=1000
gen_epochs=None
wind=500
stride=200

use_these_chanels, acc = get_selected_channel_gumbel(sid, 10)
norm_method='std'
test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid,fs,selected_channels=use_these_chanels,scaler=norm_method)
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride)
labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (304, 10, 500)


# cTGAN
ax.clear()
time_stamp='2024_04_10_16_09_06'
result_dir='D:/tmp/python/gesture/DA/cTGAN/sid' + str(sid) + '/'+time_stamp+'/'
filename=result_dir+'Samples/class_'+str(classi)+'.npy'
gen_data=np.load(filename) # (304, 10, 500)
plot_class0=[X_train_class0, gen_data]
labels=['orig','ttscgan']
fig2=visualization([plot_class0[i].transpose(0,2,1) for i in range(len(plot_class0))],'tSNE',labels)
ax.plot(gen_data[0,0,:]) # blue
ax.plot(X_train_class0[0,0,:])

# cWGANGP
ax.clear()
time_stamp='2024_04_11_15_17_12'
result_dir='D:/tmp/python/gesture/DA/CWGANGP/sid' + str(sid) + '/'+time_stamp+'/'
filename=result_dir+'Samples/class_'+str(classi)+'.npy'
gen_data2=np.load(filename) # (304, 10, 500)
plot_class0=[X_train_class0, gen_data2]
labels=['orig','cWGANGP']
fig2=visualization([plot_class0[i].transpose(0,2,1) for i in range(len(plot_class0))],'tSNE',labels)
ax.plot(gen_data2[0,0,:]) # blue
ax.plot(X_train_class0[0,0,:]) # red

filename = result_dir + 'DA/plots/' + 'tSNE_orig_NI.pdf'
fig2.savefig(filename)


