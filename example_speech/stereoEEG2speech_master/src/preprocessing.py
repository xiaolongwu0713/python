import mne
import numpy as np

import socket
if socket.gethostname() == 'LongsMac':
    mydatadir = '/Volumes/Samsung_T5/data/stereoEEG2speech_master/'
elif socket.gethostname()== 'workstation':
    mydatadir = 'H:/Long/data/stereoEEG2speech_master/'

sf_EEG=[1024,1024,1024]
for sid in [1,2,3]:
    #sid=1
    data=np.load(mydatadir + 'p'+str(sid)+'_sEEG.npy').transpose() # 1024, 500.470703125 s
    print('EEG length: '+ str(data.shape[1]/1024)+' s.')
    audio=np.load(mydatadir+'p'+str(sid)+'_audio_final.npy') # 22050, 500.4740589569161 s
    print('Audio length: ' +str(audio.shape[0]/22050) + ' s.')
    chn_names=["seeg"]*data.shape[0]
    chn_types=["seeg"]*data.shape[0]
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=1024)
    raw = mne.io.RawArray(data, info)
    raw.load_data().notch_filter(np.arange(50, 251, 50))
    raw.filter(70,170)
    final_data=raw.get_data() # (127, 512482)

    filename=mydatadir+ 'p'+str(sid)+'_sEEG_gamma.npy'
    np.save(filename,final_data)



