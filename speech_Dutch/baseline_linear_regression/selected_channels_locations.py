'''
This script will extract the 3D coordination and anatomical labels of the selected channels,
then write to the mydrive/matlab/plot_brain folder for plotting using matlab.
'''
from speech_Dutch.baseline_linear_regression.opt import selected_channels

## get the anatomical labels of the selected electrodes
if __name__=="__main__":
    import numpy as np
    import os
    from speech_Dutch.config import data_dir
    from pre_all import top_data_dir

    sids=[1,2,3,4,5,6,7,8,9,10]
    melbins=23
    #result_path = data_dir + 'baseline_LR_channel_selection/SingleWordProductionDutch/mel_' + str(melbins) + '/selected_channels/'
    result_path='/Users/xiaowu/My Drive/matlab/plot_brain/speech_Dutch/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for sid in sids:
        pt = 'sub-' + f"{sid:02d}"
        individual_path = result_path+pt+'/'
        # name
        filename=top_data_dir+'SingleWordProductionDutch/'+pt+'/ieeg/'+pt+'_task-wordProduction_channels.tsv'
        text = np.loadtxt(filename, dtype=str)
        text=text[1:,:] # the first row is the description head
        ch_index=selected_channels[str(sid)]
        selected_regions=text[ch_index,:][:,[0,5]]
        selected_ele_names = text[ch_index, 0]

        # localtion
        filename = top_data_dir + 'SingleWordProductionDutch/' + pt + '/ieeg/' + pt + '_task-wordProduction_space-ACPC_electrodes.tsv'
        text = np.loadtxt(filename, dtype=str)
        text = text[1:, :]  # the first row is the description head
        locations=[]
        for n in selected_ele_names:
            ind=np.where(text == n)
            locations.append(text[ind[0][0]])

        selected_location_file = individual_path + pt + '_locations.txt'
        with open(selected_location_file, 'w') as f:
            for l in locations:
                string = ' '.join(i for i in l)
                f.write(string)
                f.write('\n')


        selected_regions_file = individual_path + pt + '_regions.txt'
        with open(selected_regions_file, 'w') as f:
            for r in selected_regions:
                string = ','.join(i for i in r)
                f.write(string)
                f.write('\n')







