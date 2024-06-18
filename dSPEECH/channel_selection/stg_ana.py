import numpy as np
import matplotlib

from dSPEECH.config import data_dir

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1)

modality='SEEG'
sid=1 # 1/2
lam=0.05
#result_file=result_dir+'selection/stg/lambda/'+str(lami)+'/P'+str(sid)+'/result'+str(sid)+'.npy'
result_dir = data_dir + 'processed/'+modality+str(sid)+'/VAI_channel_selection/'
result_file=result_dir+'result'+str(sid)+'_stg_lam'+str(lam)+'.npy'
#r'D:\data\BaiduSyncdisk\speech_Southmead\processed\SEEG2\VAI_channel_selection\result2.npy'
result = np.load(result_file, allow_pickle=True).item()
test_acc=result['test_acc']
probs=result['probs']
raws=result['raws']
ax[0].imshow(probs)
ax[1].plot(probs[-1,:])

prob=probs[-1,:]
channel_ranks=prob.argsort() # channel index ranking ascendingly
top_n_channels=None
pro_threshold=0.8
if top_n_channels is not None: # pick top n channels
    if len(channel_ranks) < top_n_channels:
        selected_channels = np.where(prob > pro_threshold)
        selected_channels = selected_channels[0]
    else:
        selected_channels=channel_ranks[-1*top_n_channels:]
else: # all selected channels
    selected_channels=np.where(prob>pro_threshold)
    selected_channels=selected_channels[0]

file=result_dir+'selected_channels_stg_lam_'+str(lam)+'.txt'
channel_string=','.join([str(i) for i in selected_channels])
with open(file,'w') as f:
    f.write('All prob bigger than 0.99 and sorted in ascending order. If to chose top 10, just pick the last 10 elements.')
    f.write('\n')
    f.write(channel_string)
    #f.write('\n')


