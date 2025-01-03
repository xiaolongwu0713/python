opt_SingleWordProductionDutch={
'norm_mel':False,
'norm_EEG':True,

'use_the_official_tactron_with_waveglow':True,
'window_eeg':False,

# parameters of feature extraction for audio data; If window_eeg=True, these parameters will also be applied to EEG;
# If False: high gamma feature will be used directly, and the length of final features of EEG and audio will be different
'target_SR':48000, # 22050/48000
'winL':0.05,
'frameshift':0.01,

## parameter of sliding for audio feature
'win':0.1,
'history':0.1,
'stride':1, # sequence length
'use_pca':False,

'baseline_method':True,
'win_baseline':10,
'history_baseline':10,
'stride_baseline':1,


'stride_test':5,

'embed_size': 256,
'num_hiddens':256,
'num_layers':2,
'dropout':0.5,
'lr':0.0005,
'batch_size':128,
'sf_EEG':1024
}

opt_mydata={
'sf_EEG':1000,
'norm_mel':False,
'norm_EEG':True,
'mel_bins':23,
'step_size':5,
'modelOrder':4,
'winL':0.05, # feature extraction
'frameshift':0.01, # feature extraction
'win':0.05, # 25 samples
'history':0.02,
'stride':50,
'use_pca':False,
'test_shift':300,

'baseline_method':True, # average the EEG signal
'win_baseline':10,
'history_baseline':10,
'stride_baseline':2,
'stride_test':5,
}


channel_numbers=[127,127,127,115,60,127,127,54,117,122]


