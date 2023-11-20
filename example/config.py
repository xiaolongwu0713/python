import numpy as np
import torch
import mne

try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print('error happens.')
    print(err)

tmp_dir='/tmp/'
tmp_data_dir='/Users/long/Documents/data/'

# paradigm definition
# channel:  SEEG+EMG+class

default_frequency=1000

# participants details
classNum = 5
Frequencies = np.array([[2, 1000], [3, 1000], [ 4, 1000], [5, 1000], [ 7, 1000], [ 8, 1000], [ 9, 1000], [ 10, 2000],
    [13, 2000], [ 16, 2000], [ 17, 2000], [ 18, 2000], [ 19, 2000], [ 20, 1000], [ 21, 1000], [ 22, 2000], [ 23, 2000],  #[24, 2000], [ 25, 2000], [ 26, 2000],
     [  29, 2000], [ 30, 2000], [ 31, 2000], [ 32, 2000], [ 34, 2000], [ 35, 1000],
    [36, 2000], [ 37, 2000], [41, 2000]])

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12]) # mu(motor cortex)/alpha(visual cortex)
fbands.append([13, 30]) # beta
fbands.append([60, 125]) # genBandPower_znormalied.py
#fbands.append([60, 140]) # gamma: 55-85

ERD=[13,30]
ERS=[55,100]

def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')

if __name__ == "__main__":
    ks = [k for k in dir() if (k[:2] != "__" and k !='np' and not callable(globals()[k]))]
    #for k in ks:
    #    print(type(k))
    printVariables(ks)

