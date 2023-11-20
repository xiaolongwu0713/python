# choose sids that all three decoding accs are high and consistent(deepnet, gumbel, stg)

import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *
from gesture.channel_selection.utils import get_good_sids, get_good_sids_gumbel,\
    get_selected_channel_gumbel, get_selected_channel_stg

good_sids=get_good_sids()

# extract final good sids
dl_result=result_dir+'deepLearning/'
window_size=500
selection_result=result_dir+'selection/'
selection_methods=['gumbel','stg']
channels_number_in_gumbel=10
da_deepnet={}
da_gumbel={}
da_stg={}
for sid in good_sids:
    # original deepnet accuracy
    sid_result = dl_result + str(sid) + '/'
    result_file = sid_result + 'training_result_deepnet' + str(sid)+'_'+str(window_size) + '.npy'
    result = np.load(result_file, allow_pickle=True).item()
    # dict_keys(['train_losses', 'train_accs', 'val_accs', 'test_acc'])
    da_deepnet[str(sid)+'_'+str(window_size)] = result['test_acc']
    for method in selection_methods:
        if method=="gumbel":
            _, acc=get_selected_channel_gumbel(sid,channels_number_in_gumbel)
            da_gumbel[str(sid)+'_'+str(channels_number_in_gumbel)]=acc
        elif method=="stg":
            _, acc=get_selected_channel_stg(sid, top_n_channels=10, ploty=False)
            da_stg[str(sid)] = acc

fig, ax=plt.subplots()
ax.plot(da_deepnet.values())
ax.plot(da_gumbel.values())
ax.plot(da_stg.values())
ax.legend(['deepnet','gumbel','stg'],loc="lower left",bbox_to_anchor=(0,0.8,0.3,0.3),fontsize='small')
#ax.get_legend().remove()
ax.set_xticks(range(len(good_sids)))
ax.set_xticklabels(good_sids)

good_sid_gumbel=get_good_sids_gumbel()
outfile=meta_dir+'good_sids_gumbel.txt'
if os.path.exists(outfile):
    os.remove(outfile)
with open(outfile, 'w') as f:
    for item in good_sid_gumbel:
        f.write("%d\n" % item)

final_good_sids=[3,4,10,13,17,18,25,29,41] # stg perform bad on 32(80% on deepnet) and 34(80% on deepnet).
outfile=meta_dir+'final_good_sids.txt'
if os.path.exists(outfile):
    os.remove(outfile)
with open(outfile, 'w') as f:
    for item in final_good_sids:
        f.write("%d\n" % item)



