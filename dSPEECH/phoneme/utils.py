import random
import numpy as np

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
    trial_number=data.shape[0]
    trial_list = list(range(trial_number))
    train_n=int(0.6*trial_number)
    val_n = int(0.2 * trial_number)
    test_n = int(0.2 * trial_number)

    test_trails=random.sample(trial_list, test_n)
    trial_number_left=np.setdiff1d(trial_list,test_trails)

    val_trails = random.sample(trial_number_left.tolist(), val_n)
    train_trails = np.setdiff1d(trial_number_left, val_trails)
    a,b,c=data[train_trails], data[val_trails], data[test_trails]
    return [i.transpose() for i in a], [i.transpose() for i in b], [i.transpose() for i in c]



