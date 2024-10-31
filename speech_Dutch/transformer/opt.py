opt_SingleWordProductionDutch={
'sf_EEG':1024,
'norm_mel':False,
'norm_EEG':True,
'mel_bins':80,
'step_size':5,
'model_order':4,

'window_eeg':True, # # use an averaging window to slide along the extracted EEG high-gamma feature
'winL':0.05, #  size of the averaging window
'frameshift':0.01, # stride of the averaging window
'target_SR':16000, #22050; for winL of 0.05, winL*target_SR=800 is an integer. This will lead to simiar length between features of eeg and audio;
'use_the_official_tactron_with_waveglow':True,



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
opt_transformer['lr']=0.0003
opt_transformer['batch_size']=128




