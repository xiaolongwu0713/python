from gesture.channel_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel_2_steps, \
    get_selected_channel_stg
from gesture.config import  *
from scipy.io import savemat

# output good sid channels
channel_to_select = 10
selected_channels_gumbel = {}
selected_channels_stg = {}
selected_channels_gumbel_list = []
selected_channels_stg_list = []
# good sids according without selection layer
good_sids=get_good_sids()
for sid in good_sids:
    selected, acc = get_selected_channel_gumbel_2_steps(sid, channel_to_select)
    selected.sort()
    selected_M = [i + 1 for i in selected]
    selected_channels_gumbel_list.append({'sid' + str(sid): selected_M})

    selected, acc, prob_vs_epoch = get_selected_channel_stg(sid, 9999)
    selected.sort()
    selected_M = [i + 1 for i in selected]
    selected_channels_stg_list.append({'sid' + str(sid): selected_M})

selected_channels_gumbel['selected_channels_gumbel'] = selected_channels_gumbel_list
selected_channels_stg['selected_channels_stg'] = selected_channels_stg_list
gumbel_filename = meta_dir + "selected_channels_gumbel.mat"
stg_filename = meta_dir + "selected_channels_stg.mat"
savemat(gumbel_filename, selected_channels_gumbel)
savemat(stg_filename, selected_channels_stg)

#### compare the selected channels by two methods ####
selected_gumbel=[]
selected_stg=[]
selected_num_gumbel=[]
selected_num_stg=[]
for i, sid in enumerate(good_sids):
    key='sid'+str(sid)
    list1=selected_channels_gumbel_list[i][key]
    list2=selected_channels_stg_list[i][key]
    selected_gumbel.append(list1)
    selected_num_gumbel.append(len(list1))
    selected_stg.append(list2)
    selected_num_stg.append(len(list2))
    print(list2)



# good sid according to both gumbel and stg
final_good_sids=get_final_good_sids()
selected_channels_gumbel_list = []
selected_channels_stg_list = []
# output final_good_sid channels
for sid in final_good_sids:
    selected, acc = get_selected_channel_gumbel_2_steps(sid, channel_to_select, ploty=False)
    selected.sort()
    selected_M = [i + 1 for i in selected]
    selected_channels_gumbel_list.append({'sid' + str(sid): selected_M})

    selected, acc = get_selected_channel_stg(sid, channel_to_select, ploty=False)
    selected.sort()
    selected_M = [i + 1 for i in selected]
    selected_channels_stg_list.append({'sid' + str(sid): selected_M})

selected_channels_gumbel['selected_channels_gumbel'] = selected_channels_gumbel_list
selected_channels_stg['selected_channels_stg'] = selected_channels_stg_list
gumbel_filename = meta_dir + "selected_channels_gumbel_final.mat"
stg_filename = meta_dir + "selected_channels_stg_final.mat"
savemat(gumbel_filename, selected_channels_gumbel)
savemat(stg_filename, selected_channels_stg)




