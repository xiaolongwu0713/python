'''
No necessary, one can just extract and work on the speaking part
'''

from dSPEECH.config import *
from scipy.io import wavfile
modality='SEEG'
sid=2
sf=1024
result_dir = data_dir + 'processed/'+modality+str(sid)+'/VAD/'
## load epochs and sentences
filename=data_dir+'processed/'+modality+str(sid)+'/'+modality+str(sid)+'-epo.fif'
epochs=mne.read_epochs(filename)
filename2=data_dir+'processed/'+modality+str(sid)+'/sentences.npy'
sentences=np.load(filename2,allow_pickle=True)

data=epochs.get_data()
eeg=data[:,:,5*sf:10*sf] # (100, 149, 5120)
audio_file=data_dir+'/paradigm_audio/5seconds_combined.wav'
audio_sr, audio = wavfile.read(audio_file)




