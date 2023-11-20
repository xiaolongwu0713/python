import os, re
import socket
import pre_all # add python project path

import sys
#print('Python%son%s'%(sys.version,sys.platform))
if socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
elif socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])

sids=[1,2,3,4,5,6,7,8,9]

if socket.gethostname() == 'ScottsMachine': # change this to your $HOSTNAME
    data_dir = '/your/data/dir/'  # running on Scott's machine

location=os.getcwd()
if re.compile('/Users/long/').match(location): # long's machine
    data_dir='/Users/long/Documents/data/ecog/'
    tmp_data_dir='/Users/long/Documents/data/gesture/'
    root_dir = '/Users/long/BCI/python_scripts/googleDrive/'
elif re.compile('/content/drive').match(location):  # googleDrive
    data_dir='/content/drive/MyDrive/data/ecog/'
    root_dir='/content/drive/MyDrive/'
elif re.compile('C:').match(location): # windows workstation
    data_dir='C:/Users/wuxiaolong/Desktop/BCI/data/ecog/'
    root_dir='C:/Users/wuxiaolong/Desktop/BCI/googledrive/'
