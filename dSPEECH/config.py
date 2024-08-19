from pre_all import *

data_dir = top_data_dir+'speech_Southmead/'
meta_dir=top_meta_dir+'dSPEECH/'
result_dir = data_dir + 'result/'
ele_dir = data_dir + 'EleCTX_Files/'
info_dir = data_dir + 'info/'


opt_regression_LR={
'model_name':'seq2seq_transformer',
'melbins':23,
'stepsize':1,
'modelorder':4, # Regression result using 4 is similar to 6.
'winlen':0.05, # feature extraction
'frameshift':0.01, # feature extraction
}

