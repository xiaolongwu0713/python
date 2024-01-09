opt_SingleWordProductionDutch={
'sf_EEG':1024,
'norm_mel':False,
'norm_EEG':True,
'mel_bins':80,
'step_size':5,
'model_order':4,

'window_eeg':False,
'target_SR':22050,
'use_the_official_tactron_with_waveglow':True,
'winL':0.05, # feature extraction
'frameshift':0.01, # feature extraction


'win':0.1, # 10 samples
'history':0.1,
'stride':1, # training stride
'use_pca':False,

'baseline_method':False, # average the EEG signal
'win_baseline':10,
'history_baseline':10,
'stride_baseline':2,

'stride_test':5,
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

opt_transformer={}
opt_transformer['Transformer-layers'] = 3 # 6
opt_transformer['Model-dimensions'] = 256 # 256
opt_transformer['feedford-size'] = 256 # 512
opt_transformer['headers'] = 4 # 8
opt_transformer['dropout'] = 0.5
opt_transformer['lr']=0.0005
opt_transformer['batch_size']=128




