import sys,os
sys.dont_write_bytecode = True

import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42

gettrace = getattr(sys, 'gettrace', None)
if gettrace is None:
    debugging=False
elif gettrace():
    debugging=True
else:
    debugging=False

import socket
driver='mydrive' # 'OneDrive/mydrive  #
if socket.gethostname() == 'LongsMac': # or laptop
    #sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    if os.path.exists('/Volumes/Samsung_T5/data/'):
        top_data_dir='/Volumes/Samsung_T5/data/'
    else:
        top_data_dir = '/Volumes/second/data_local/' #'/Users/long/Documents/data/'# temp data dir
    #tmp_data_dir='/Users/long/Documents/data/gesture/'
    top_root_dir = '/Users/long/'+driver+'/python/'  # this is project root on google drive
    top_meta_dir = '/Users/long/'+driver+'/meta/'
    computer='mac'
elif socket.gethostname() == 'Long': # Yoga
    # sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    top_data_dir = 'D:/data/'
    # tmp_data_dir='/Users/long/Documents/data/gesture/'
    top_root_dir = 'C:/Users/xiaowu/' + driver + '/python/'
    top_meta_dir = 'C:/Users/xiaowu/' + driver + '/meta/'
    computer = 'Yoga'
elif socket.gethostname() == 'DESKTOP-FBDP919': # or laptop
    #sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    top_data_dir = 'G:/data/'
    #tmp_data_dir='/Users/long/Documents/data/gesture/'
    top_root_dir = 'C:/Users/xiaowu/' + driver + '/python/'
    top_meta_dir = 'C:/Users/xiaowu/' + driver + '/meta/'
    computer='mac'
elif socket.gethostname() == 'workstation':
    #sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
    #data_dir = 'C:/Users/wuxiaolong/Desktop/BCI/data/gesture/'  # temp data dir
    top_data_dir = 'H:/Long/data/'  # temp data dir
    top_root_dir='C:/Users/wuxiaolong/'+driver+'/python/'
    top_meta_dir = 'C:/Users/wuxiaolong/'+driver+'/meta/'
    tmp_dir = 'H:/Long/data/tmp_dir_python/'
    computer='workstation'
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    top_data_dir = 'H:/Long/data/'  # temp data dir
    top_root_dir = 'C:/Users/xiaol/'+driver+'/python/'
    top_meta_dir = 'C:/Users/xiaol/'+driver+'/meta/'
    tmp_dir='H:/Long/data/tmp_dir_python/'
elif socket.gethostname() == 'LongsPC':
    top_data_dir = 'H:/Long/data/'  # temp data dir
    top_root_dir = 'C:/Users/Long/'+driver+'/python/'
    top_meta_dir = 'C:/Users/Long/'+driver+'/meta/'

import os, re
location=os.getcwd()
if re.compile('/content/drive').match(location):  # googleDrive
    top_data_dir='/content/drive/MyDrive/data/'
    top_root_dir='/content/drive/MyDrive/' # googleDrive
    info_dir = '/content/drive/MyDrive/data/'
    top_meta_dir = '/content/drive/MyDrive/data/'
    computer = 'google'

colors=['orangered','skyblue','orange','springgreen','aquamarine','yellow','gold']

import os

# if 'PYTHONPATH' in os.environ and 'PyCharm' in os.environ['PYTHONPATH']:
if os.environ.get('PYCHARM_HOSTED'):
    running_from_IDE = True
    running_from_CMD = False
    print("pre_all: Running from IDE.")
else:
    running_from_CMD = True
    running_from_IDE = False
    print("pre_all: Running from CMD.")

from common_dl import *
from comm_utils import *
from common_plot import *
