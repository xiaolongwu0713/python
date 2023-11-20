
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *

# good sid
def get_good_sids():
    outfile=meta_dir+'good_sids.txt'
    good_sids=[]
    with open(outfile, 'r') as f:
        for line in f:
            good_sids.append(int(line.rstrip('\n')))
    return good_sids

# final good sid
def get_final_good_sids():
    outfile=meta_dir+'final_good_sids.txt'
    final_good_sids=[]
    with open(outfile, 'r') as f:
        for line in f:
            final_good_sids.append(int(line.rstrip('\n')))
    return final_good_sids


def get_selected_channel_gumbel_2_steps(sid, channel_num_selected):
    channel_num_selected=10
    selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/' + str(channel_num_selected) + 'channels/'
    score_all = np.load(selection_result + 'score_all.npy', allow_pickle=True).item() # (train acc, val acc)
    h = np.load(selection_result + 'HH.npy')
    s = np.load(selection_result + 'SS.npy')  # selection
    z = np.load(selection_result + 'ZZ.npy')  # probability
    h = np.squeeze(h)
    z = np.squeeze(z)
    test_acc = score_all['test_acc'].numpy()

    selected_channels = np.argmax(z[-1, :, :], axis=0)
    selected_channels = list(set(selected_channels))

    return selected_channels, test_acc

def get_good_sids_gumbel():
    info = info_dir + 'info.npy'
    info = np.load(info, allow_pickle=True)
    sids = info[:, 0]
    acc_all=[]
    for i, sid in enumerate(sids):
        acc_all.append([])
        selected_channels, test_acc=get_selected_channel_gumbel_2_steps(sid,10)
        acc_all[i]=test_acc
    acc_all=np.asarray(acc_all)
    return sids[acc_all>0.5]

def get_selected_channel_gumbel(sid, channel_num_selected, ploty=False):
    selection_result = result_dir + 'selection/gumbel/' + 'P' + str(sid) + '/channels' + str(channel_num_selected) + '/'
    scores = np.load(selection_result + 'epoch_scores.npy')  # (train acc, val acc)
    h = np.load(selection_result + 'HH.npy')
    s = np.load(selection_result + 'SS.npy')  # selection
    z = np.load(selection_result + 'ZZ.npy')  # probability
    h = np.squeeze(h)
    z = np.squeeze(z)

    mean_entropy = np.mean(h, axis=1)
    # best training epoch: how to find the best epoch: lowest entropy + highest val acc
    highest_acc = max(scores[:, 1])
    best_evaluated = np.where(scores[:, 1] == highest_acc)[0][0]

    if ploty:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(scores[:, 0])
        ax[0].plot(scores[:, 1])
        ax[0].plot(mean_entropy)
        ax[0].legend(['train acc', 'val acc', 'entropy'], loc="lower left", bbox_to_anchor=(0, 0.8, 0.1, 0.1),
                     fontsize='small')
        # plot matrix
        ax[1].imshow(z[best_evaluated, :, :])

    selected_channels = np.argmax(z[best_evaluated, :, :], axis=0)
    selected_channels = list(set(selected_channels))
    if sid==17:
        selected_channels.append(19) # chosen ch 27 twice;
    result_file = selection_result +'test_acc.npy'
    from os.path import exists
    if exists(result_file):
        test_acc = np.load(result_file, allow_pickle=True).item()
    else:
        test_acc=9999999
    return selected_channels, test_acc

def  get_selected_channel_stg(sid, top_n_channels=None):
    selection_result = result_dir + 'selection/stg/0.2/'
    result_file=selection_result+'P'+str(sid)+'/result'+str(sid)+'.npy'
    result = np.load(result_file, allow_pickle=True).item()
    test_acc=result['test_acc']
    if 1==0:
        best_epoch=result['validate_acc'].argsort()[-1]
    else:
        # or best_epoch is the largest epoch
        best_epoch=-1
    probs=result['probs'][best_epoch,:]
    prob_vs_epoch=result['probs'] # for im 2D plot
    channel_ranks=probs.argsort() # channel index ranking ascendly
    if top_n_channels: # pick top n channels
        if len(channel_ranks) < top_n_channels:
            selected_channels = np.where(probs > 0.99)
            selected_channels = selected_channels[0]
        else:
            selected_channels=channel_ranks[-1*top_n_channels:]
    else: # all selected channels
        selected_channels=np.where(probs>0.99)
        selected_channels=selected_channels[0]
    return selected_channels, test_acc, prob_vs_epoch

