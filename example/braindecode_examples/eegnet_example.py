'''
this is a working workflow to study EEGNet model.
'''
import mne
from braindecode.datautil import (create_from_mne_raw, create_from_mne_epochs)
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv1, EEGNetv4
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

###############################################################################
# First, fetch some data using mne:

# 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
subject_id = 22
event_codes = [5, 6, 9, 10, 13, 14]
# event_codes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes, update_path=False)

# Load each of the files
parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')for path in physionet_paths]

###############################################################################
# Convert mne.RawArrays to a compatible data format:
descriptions = [{"event_code": code, "subject": subject_id}for code in event_codes]
windows_datasets = create_from_mne_raw(
    parts,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=500,
    window_stride_samples=500,
    drop_last_window=False,
    descriptions=descriptions,
)

###############################################################################
# If trials were already cut beforehand and are available as mne.Epochs:

# list_of_epochs = [mne.Epochs(raw, [[0, 0, 0]], tmin=0, baseline=None)
#                   for raw in parts]
# windows_datasets = create_from_mne_epochs(
#     list_of_epochs,
#     window_size_samples=50,
#     window_stride_samples=50,
#     drop_last_window=False
# )

desc=windows_datasets.description.rename(columns={'subject': 'split'})
# pusdo split some train and test dataset
desc.iloc[0:3,1]='train'
desc.iloc[3:,1]='test'
windows_datasets.description=desc
splitted = windows_datasets.split('split')
train_set = splitted['train']
valid_set = splitted['test']

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 3
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

#model = ShallowFBCSPNet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
model = EEGNetv1(n_chans,n_classes,input_window_samples)
# Send model to GPU
if cuda:
    model.cuda()

#x = torch.randn(1, n_chans, input_window_samples)  # input: torch.Size([64, 64, 500])
#y=model(x)


# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)
