## Used in GAN: generate the artifical data of certain shape
# To get channel number for each participant
from gesture.config import *
import hdf5storage
from gesture.utils import read_sids

sids=read_sids()
channel_number={}
for sid in sids:
    data_folder = data_dir + 'preprocessing/' + 'P' + str(sid) + '/'
    data_path = data_folder + 'preprocessing2.mat'
    mat = hdf5storage.loadmat(data_path)
    data = mat['Datacell']
    good_channels = mat['good_channels']
    print(len(np.squeeze(good_channels)))
    channel_number[str(sid)] = len(np.squeeze(good_channels))

filename=meta_dir+'info/channel_number.txt'
if not os.path.exists(filename):
    os.makedirs(filename)

with open(filename,"w") as f:
    for key in channel_number.keys():
        astring=key+','+str(channel_number[key])
        f.write(astring)
        f.write('\n')
















