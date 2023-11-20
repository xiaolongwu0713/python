selected_channels={
'1':[56,49,6,91,79,47,84,46,67,26,86,58,55,22,51],
'2':[108,101,116,26,57,84,15,48,81,79,89,119,120,87,27],
'3':[48,47,87,97,63,113,9,111,106,41,70,99,42,53,45],
'4':[30,20,36,16,7,108,112,102,54,100,0,4,84,97,86],
'5':[47,13,24,44,40,37,54,45,31,39,58,11,28,41,38],
'6':[30,43,35,47,52,98,61,22,48,102,50,8,74,51,100],
'7':[8,100,51,31,74,85,50,52,101,25,94,11,0,102,30],
'8':[47,7,31,3,13,26,0,29,22,46,9,18,35,49,12],
'9':[89,26,116,9,90,81,112,108,98,80,109,39,110,114,87],
'10':[33,37,61,55,39,112,0,35,57,38,56,58,97,60,53],
}

if __name__=="__main__":
    import numpy as np
    import os
    from speech_Ruijin.config import data_dir
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







