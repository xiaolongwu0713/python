## A timer to calculate the running period
from utils import my_timer
## Indicate the name of the script file
#import __main__ as main
#print('Running '+ main.__file__+'.') # only works when execute in CMD, but not IDE
## TODO: what's this?
import sys,os
sys.dont_write_bytecode = True

gettrace = getattr(sys, 'gettrace', None)
if gettrace is None:
    debugging=False
elif gettrace():
    debugging=True
else:
    debugging=False

import socket
drive='mydrive' # 'OneDrive/mydrive  #
if socket.gethostname() == 'LongsMac': # or laptop
    #sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    if os.path.exists('/Volumes/Samsung_T5/data/'):
        top_data_dir='/Volumes/Samsung_T5/data/'
    else:
        top_data_dir = '/Volumes/second/data_local/' #'/Users/long/Documents/data/'# temp data dir
    #tmp_data_dir='/Users/long/Documents/data/gesture/'
    mydrive='/Users/xiaowu/'+drive+'/'
    top_root_dir = '/Users/xiaowu/'+drive+'/python/'  # this is project root on google drive
    top_meta_dir = '/Users/xiaowu/'+drive+'/meta/'
    tmp_dir = '/Users/xiaowu/tmp/python_log/'
    computer='mac'
elif socket.gethostname() == 'Long': # Yoga
    # sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    top_data_dir = 'D:/data/BaiduSyncdisk/'
    top_root_dir = 'D:/' + drive + '/python/'
    top_meta_dir = 'D:/' + drive + '/meta/'
    # tmp_data_dir='/Users/long/Documents/data/gesture/'
    mydrive='D:/' + drive+'/'
    computer = 'Yoga'
    tmp_dir='D:/tmp/python/'

elif socket.gethostname() == 'workstation':
    #sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
    #data_dir = 'C:/Users/wuxiaolong/Desktop/BCI/data/gesture/'  # temp data dir
    top_data_dir = 'H:/Long/data/'  # temp data dir
    top_root_dir='C:/Users/wuxiaolong/'+drive+'/python/'
    top_meta_dir = 'C:/Users/wuxiaolong/'+drive+'/meta/'
    tmp_dir = 'H:/Long/data/tmp_dir_python/'
    computer='workstation'

elif socket.gethostname() == 'LongsPC':
    top_data_dir = 'H:/Long/data/'  # temp data dir
    top_root_dir = 'C:/Users/Long/'+drive+'/python/'
    top_meta_dir = 'C:/Users/Long/'+drive+'/meta/'

import os, re
location=os.getcwd()
if re.compile('/content/drive').match(location):  # googleDrive
    top_data_dir='/content/drive/MyDrive/data/'
    top_root_dir='/content/drive/MyDrive/' # googleDrive
    info_dir = '/content/drive/MyDrive/data/'
    top_meta_dir = '/content/drive/MyDrive/data/'
    computer = 'google'

## try to log everything
from datetime import datetime
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
logfilename=tmp_dir+'log/pycharm/'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'_log.txt'
sys.stdout = Logger(filename=logfilename)

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
