import numpy as np
import mne
try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print('error happens.')
    print(err)

# data is an epoch data( after .get_data() call.)
def slide_epochs(epoch, label, wind, stride):
    total_len = epoch.shape[2]
    X = []
    Xi = []
    for trial in epoch:  # (63, 2001)
        s = 0
        while stride * s + wind < total_len:
            start = s * stride
            tmp = trial[:, start:(start + wind)]
            Xi.append(tmp)
            s = s + 1
        # add the last window
        last_s = s - 1
        if stride * last_s + wind < total_len - 100:
            tmp = trial[:, -wind:]
            Xi.append(tmp)

    X.append(Xi)
    X = np.concatenate(X, axis=0)  # (1300, 63, 500)
    samples_number = len(X)
    y = [label] * samples_number

    return X, y

# smooth a 1-D time series y using convolutional window
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


