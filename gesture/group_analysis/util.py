from gesture.config import *
from gesture.utils import get_epoch, get_good_channels

filename=meta_dir+'ele_anat_position.npy'
ele = np.load(filename,allow_pickle=True).item()
def reduce_duplicated_hyph(aname):
    aname2 = []
    prev = aname[0]
    for c in aname[1:]:
        if c == '-':
            if prev == '-':
                pass
            else:
                aname2.append(prev)
        else:
            aname2.append(prev)
        prev = c
    aname2.append(c)
    aname2 = ''.join(i for i in aname2)
    return aname2


def calculate_CC(sid1, sf1, sid2, sf2, f1,f2,trigger='EMG',re_ordered=True,by_channel=True,random_shift=False,duration=0.5):
    datas = []  # trial, channel, time
    #total_ch_numbers = []
    #good_channels = []
    #bad_channels_inds = []
    #bad_channels = []
    ordered_ch_names = []
    for sid,sf in zip([sid1, sid2], [sf1, sf2]):
        epoch_ordered= load_data_and_clean_up(sid, sf, duration=duration, f1=f1,f2=f2,trigger=trigger, re_ordered=re_ordered,random_shift=random_shift)
        datas.append(epoch_ordered.pick(picks=['eeg']).get_data())
        # epochs_tmp = get_epoch(sid, sf, scaler='no', trigger='EMG', tmin=tmin, tmax=tmax)
        # epoch = epochs_tmp['3']  # simple grasp
        #
        # total_ch_number = len(epoch.info.ch_names) #128
        # #total_ch_numbers.append(total_ch_number)
        # good_channel = good_channel_dict['sid' + str(sid)]  # start from 1 (Matlab)
        # good_channel = [i - 1 for i in good_channel] # 115
        # #good_channels.append(good_channel)
        # bad_channels_ind = [i for i in range(len(epoch.ch_names) - 7) if i not in good_channel] # 6
        # #bad_channels_inds.append(bad_channels_ind)
        # bad_channel = [epoch.ch_names[c] for c in bad_channels_ind]
        # #bad_channels.append(bad_channel)
        # epoch.load_data()
        # epoch.drop_channels(bad_channel)
        #
        # info_tmp = epoch.info
        # old = info_tmp.ch_names # 122=128-6
        # new = ele['sid' + str(sid)]['ana_label_id'] # 115
        # mapping = {old[i]: new[i] for i in range(len(new))}
        # mne.rename_channels(info_tmp, mapping)
        # epoch.info = info_tmp
        #
        # assert len(good_channel) == len(new)
        # assert total_ch_number == len(new) + len(bad_channel) + 7
        #
        # ch_names = epoch.info.ch_names
        # sorted_index = sorted(range(len(ch_names)), key=ch_names.__getitem__)
        # ordered_ch_name = [ch_names[c] for c in sorted_index]
        # ordered_ch_names.append(ordered_ch_name)
        # epoch_ordered = epoch.reorder_channels(ordered_ch_name)
        # datas.append(epoch_ordered.pick(picks=['eeg']).get_data())

    if by_channel:
        matrixes=np.zeros([datas[0].shape[1],datas[1].shape[1]]) # shape: channel number of a * channel number of b
        for ch_a in range(datas[0].shape[1]):
            a = datas[0][:, ch_a, :]  # trial, time
            for ch_b in range(datas[1].shape[1]):
                b = datas[1][:, ch_b, :]

                column = []
                for trial_a in range(a.shape[0]):
                    row = []
                    for trial_b in range(b.shape[0]):
                        cc = np.corrcoef(a[trial_a, :], b[trial_b, :])
                        row.append(cc[0, 1])
                    column.append(sum(row)/len(row))
                avg=sum(column)/len(column)
                matrixes[ch_a,ch_b]=avg
    else: # by trial
        matrixes = []
        for trial in range(datas[0].shape[0]):
            a = datas[0][trial]  # channel, time
            b = datas[1][trial]
            matrix = []
            for ca in range(a.shape[0]):
                row = []
                for cb in range(b.shape[0]):
                    cc = np.corrcoef(a[ca, :], b[cb, :])
                    row.append(cc[0,1])
                matrix.append(row)
            matrixes.append(matrix)
        matrixes=np.asarray(matrixes)
    return matrixes, ordered_ch_names

# read the data--> bandpass(f1,f2)-->drop bad channels-->rename the channel names (anatomic names)---> re-order channel according to the locations
def load_data_and_clean_up(sid, sf,duration=0.5, f1=0.1,f2=4,trigger='EMG',re_ordered=True,random_shift=False):
    tmin = 0
    tmax = duration

    good_channel_dict = get_good_channels()
    epochs_tmp = get_epoch(sid, sf, scaler='no', trigger=trigger, tmin=tmin, tmax=tmax,random_shift=random_shift)
    epoch = epochs_tmp['3'].load_data().filter(l_freq=f1, h_freq=f2, picks=['eeg'])  # simple grasp
    type_list=epoch.info.get_channel_types()
    non_eeg=sum([i.upper()!='EEG' for i in type_list])
    total_ch_number = len(epoch.info.ch_names)  # 128
    good_channel = good_channel_dict['sid' + str(sid)]  # start from 1 (Matlab)
    good_channel = [i - 1 for i in good_channel]  # 115
    bad_channels_ind = [i for i in range(len(epoch.ch_names) - non_eeg) if i not in good_channel]  # 7
    bad_channel = [epoch.ch_names[c] for c in bad_channels_ind]
    epoch.load_data()
    epoch.drop_channels(bad_channel)

    # tmp deactivate
    if False:
        info_tmp = epoch.info
        old = info_tmp.ch_names  # 122=128-6
        new = ele['sid' + str(sid)]['ana_label_id']  # 115
        mapping = {old[i]: new[i] for i in range(len(new))}
        mne.rename_channels(info_tmp, mapping)
        epoch.info = info_tmp

        assert len(good_channel) == len(new)
        assert total_ch_number == len(new) + len(bad_channel) + 7

    ch_names = epoch.info.ch_names
    sorted_index = sorted(range(len(ch_names)), key=ch_names.__getitem__)
    ordered_ch_name = [ch_names[c] for c in sorted_index]
    #ordered_ch_names.append(ordered_ch_name)
    epoch_ordered = epoch.reorder_channels(ordered_ch_name)
    #datas.append(epoch_ordered.pick(picks=['eeg']).get_data())
    if re_ordered:
        return epoch_ordered
    else:
        return epoch