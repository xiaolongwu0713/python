import numpy as np
import torch
import mne
try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print('error happens.')
    print(err)

tmp_dir='/tmp/'

import os, re
location=os.getcwd()
if re.compile('/Users/long/').match(location):
    data_dir='/Volumes/Samsung_T5/data/ruijin/MI/RJ_MI_Raw_Data/' # preprocessed data
    root_dir = '/Users/long/BCI/python_scripts/googleDrive/ruijin_MI/'  # this is project root
elif re.compile('/content/drive').match(location):
    data_dir='/content/drive/MyDrive/data/ruijin/MI/RJ_MI_Raw_Data/' # googleDrive
    root_dir='/content/drive/MyDrive/ruijin_MI/' # googleDrive
processed_data=data_dir

paradigm=1

default_frequency=1000

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12]) # mu(motor cortex)/alpha(visual cortex)
fbands.append([13, 30]) # beta
fbands.append([60, 125]) # genBandPower_znormalied.py
#fbands.append([60, 140]) # gamma: 55-85

ERD=[13,30]
ERS=[55,100]
# some cross module variables, you can import this variable as:
# import grasp.config as myVar, then myVar.preds=...
# OR, just make them global
#preds=[]
#targets=[]

# Lambda is used by skorch get_loss function
#Lambda = 1e-6
preds=[]
targets=[]

def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')

if __name__ == "__main__":
    ks = [k for k in dir() if (k[:2] != "__" and k !='np' and not callable(globals()[k]))]
    #for k in ks:
    #    print(type(k))
    printVariables(ks)

