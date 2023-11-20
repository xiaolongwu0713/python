opt_SingleWordProductionDutch={
'batch_size':256,
'wind_size':10, # 200-->400
'stride':5,
'epoch_num':300,
'patients':20,
'mel_bins':23,
'sf_EEG':1024,
'learning_rate':0.005, # resnet:0.01,
'dropout':0.5,
'net_name':'resnet', #'resnet'/tsception
}


opt_mydata={
'batch_size':200,
'wind_size':200,
'stride':3,
'epoch_num':300,
'patients':20,
}