opt_SingleWordProductionDutch={
'norm_mel':False,
'norm_EEG':True,
'mel_bins':23,
'target_SR':22050,
'use_the_official_tactron_with_waveglow':False,
'winL':0.05, # feature extraction
'frameshift':0.01, # feature extraction
'win':0.1,
'history':0.1,
'stride':1,
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

channel_numbers=[127,127,127,115,60,127,127,54,117,122]


