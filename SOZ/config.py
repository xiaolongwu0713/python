import numpy as np
import mne
import os, re,sys
import socket
if socket.gethostname() == 'workstation':
    mydrive_dir='C:/Users/wuxiaolong/mydrive/'
elif socket.gethostname() == 'LongsMac':
    mydrive_dir='/Users/long/My Drive/'
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    mydrive_dir='C:/Users/xiaol/My Drive/'
try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print('error happens.')
    print(err)

tmp_dir='/tmp/'
driver='mydrive' # 'OneDrive/mydrive
project_name='SOZ'
if socket.gethostname() == 'LongsMac': # or laptop
    #sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    if os.path.exists('/Volumes/Samsung_T5/data/gesture/'):
        data_dir='/Volumes/Samsung_T5/data/gesture/'
    else:
        data_dir = '/Users/long/Documents/data/'+project_name+'/'# temp data dir
    #tmp_data_dir='/Users/long/Documents/data/gesture/'
    root_dir = '/Users/long/'+driver+'/python/'+project_name+'/'  # this is project root on google drive
    result_dir = root_dir + 'result/'

    meta_dir = '/Users/long/'+driver+'/meta/'+project_name+'/'
    ele_dir = meta_dir + 'EleCTX_Files/'
    info_dir = meta_dir + 'info/'  # sub info

elif socket.gethostname() == 'workstation':
    #sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
    #data_dir = 'C:/Users/wuxiaolong/Desktop/BCI/data/gesture/'  # temp data dir
    data_dir = 'H:/Long/data/'+project_name+'/'  # temp data dir
    root_dir='C:/Users/wuxiaolong/'+driver+'/python/'+project_name+'/'
    result_dir = root_dir + 'result/'

    meta_dir = 'C:/Users/wuxiaolong/'+driver+'/meta/'+project_name+'/'
    ele_dir=meta_dir+'EleCTX_Files/'
    info_dir = meta_dir+'info/'

elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    data_dir = 'H:/Long/data/'+project_name+'/'  # temp data dir
    root_dir = 'C:/Users/xiaol/'+driver+'/python/'+project_name+'/'
    result_dir = root_dir + 'result/'

    meta_dir = 'C:/Users/xiaol/'+driver+'/meta/'+project_name+'/'
    ele_dir = meta_dir + 'EleCTX_Files/'
    info_dir = meta_dir + 'info/'

location=os.getcwd()
if re.compile('/content/drive').match(location):  # googleDrive
    data_dir='/content/drive/MyDrive/data/gesture/'
    root_dir='/content/drive/MyDrive/' # googleDrive
    info_dir = '/content/drive/MyDrive/data/gesture/preprocessing/'
colors=['orangered','skyblue','orange','springgreen','aquamarine','yellow','gold']
# paradigm definition
# channel:  SEEG+EMG+class

default_frequency=1000

soz_channel_all={
    'DHM': ['POL B1', 'POL B2', 'POL B3', 'POL B4','EEG C1-Ref', 'EEG C2-Ref', 'EEG C3-Ref', 'EEG C4-Ref', 'EEG C5-Ref'],
    'FXL': [],
    'HJY': ['EEG A1-Ref', 'EEG A2-Ref', 'POL A3', 'POL A4', 'POL A5', 'POL A6', 'POL B1', 'POL B2', 'POL B3', 'POL B4',
            'EEG C1-Ref', 'EEG C2-Ref', 'EEG C3-Ref', 'EEG C4-Ref',],
    'JXC': [],
    'LCY': [],
    'LDX': [],
    'LXL': [],
    'LY' : [],
    'SDF': [],
    'SGH': [],
    'SQL': [],
    'WXG': [],
    'XY' : [],
    'ZGQ': [],
    'ZPP': []
}
# HJY: A1-6,B1-4,C1-4

participants=list(soz_channel_all.keys())

exclude_channels_all={
    'DHM':['POL DC09', 'POL DC10', 'POL DC11', 'POL DC12', 'POL DC13', 'POL DC14', 'POL DC15',
           'POL DC16', 'POL E', 'POL EKG1', 'POL EKG2', 'POL EMGL1', 'POL EMGL2', 'POL EMGR1', 'POL EMGR2',
           'POL BP1', 'POL BP2', 'POL BP3', 'POL BP4'],
    'FXL': [],
    'HJY': ['POL DC16', 'POL EKG1', 'POL EKG2', 'POL EMGL1', 'POL EMGR1', 'POL EMGL2', 'POL EMGR2', 'POL BP1',
            'POL BP2', 'POL BP3', 'POL BP4'],
    'JXC': [],
    'LCY': [],
    'LDX': [],
    'LXL': [],
    'LY': [],
    'SDF': [],
    'SGH': [],
    'SQL': [],
    'WXG': [],
    'XY': [],
    'ZGQ': [],
    'ZPP': []
}

noisy_channels_all={
    'DHM':[],
    'FXL': [],
    'HJY': ['EEG A1-Ref', 'POL E'],
    'JXC': [],
    'LCY': [],
    'LDX': [],
    'LXL': [],
    'LY': [],
    'SDF': [],
    'SGH': [],
    'SQL': [],
    'WXG': [],
    'XY': [],
    'ZGQ': [],
    'ZPP': []
}
