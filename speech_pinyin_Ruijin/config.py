from pre_all import *

data_dir = top_data_dir+'speech_pinyin_Ruijin/'
result_dir = data_dir + 'result/'
meta_dir = data_dir #top_meta_dir+'speech_RuiJin/'
ele_dir = data_dir + 'EleCTX_Files/'
info_dir = data_dir + 'info/'

audio_sr=48000
sf_EEG=1000 # Hz
sf_audio=48000 # Hz

# pause happens right after this trial index
pause={}
pause['1']=[[57, 114],[63],[45, 107, 124]] # three sessions, each session have different number of pausing





