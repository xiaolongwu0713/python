import random
import numpy as np
from dSPEECH.config import meta_dir
import pandas

# parse the paradigm excel file into list (sentence) of list (syllable) of list (phoneme).
#This information can be used for downstream analysis, such as compare with Gentle output, and some sanity checking.
def parse_para_excel():
    filename = meta_dir + 'sentences/sentences_v3.xlsx'
    tmp = pandas.read_excel(filename)
    sentences_para = tmp.to_numpy()[0:-1, 1:9]

    result = []
    for sentence in sentences_para:
        phone_sent = []
        for syllable in sentence:
            if isinstance(syllable, float):
                phone_sent.append('nan')
            else:
                phone_sent.append(syllable.split('|')[1:-1])
        result.append(phone_sent)
    return result
def wind_list_of_2D(ons):
    ons_wind=[]
    win=100
    stride=20
    discard=0
    for on in ons:
        lens=on.shape[1]
        if lens>=120: # 150 ms
            # windowing operation
            i=0
            start=int(i*stride)
            end=start+100
            while end<lens:
                ons_wind.append(on[:,start:end])
                i=i+1
                start=int(i*stride)
                end=start+100
        elif 100<lens<120:
            ons_wind.append(on[:,:100])
            ons_wind.append(on[:,-100:])
        else:
            discard=discard+1
    return ons_wind

def train_test_split(data):
    trial_number=len(data)
    trial_list = list(range(trial_number))
    train_n=int(0.6*trial_number)
    val_n = int(0.2 * trial_number)
    test_n = int(0.2 * trial_number)

    test_trails=random.sample(trial_list, test_n)
    trial_number_left=np.setdiff1d(trial_list,test_trails)

    val_trails = random.sample(trial_number_left.tolist(), val_n)
    train_trails = np.setdiff1d(trial_number_left, val_trails)
    a,b,c=[data[i] for i in train_trails.tolist()], [data[i] for i in val_trails],[data[i] for i in test_trails]
    return [i.transpose() for i in a], [i.transpose() for i in b], [i.transpose() for i in c]


