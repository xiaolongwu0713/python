import numpy as np
from speech_Dutch.utils import TacotronSTFT, fold_2d23d2


from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from speech_Dutch.config import *
from speech_Dutch.utils import fold_2d23d

sids=[1,2]
sessions=2
sf_EEG=1000
scaler = StandardScaler()

'''
# exam the audio clip
import sounddevice as sd
fs=48000
clampi=np.clip(pairs_bad[0][0], 3, 6)
sd.play(pairs_bad[0][0],fs)
sd.play(clampi,fs)
sd.stop()
'''
# length_ration: EEG/mel=5.29
length_ration=5.29
mel_size=80
def dataset_seq2seq(sid,mel_bins=40,pre_EEG=100,post_EEG=100,window_size=None,stride=None,highgamma=True):#window_size is the EEG window size,not the audio
    mel_transformer = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=mel_bins,
                                   sampling_rate=48000, mel_fmin=0.0, mel_fmax=8000.0)
    assert pre_EEG <= extra_EEG_pairring*sf_EEG and post_EEG <= extra_EEG_pairring*sf_EEG
    SOS=0.1
    z=[] #highgamma
    x=[] # raw
    y=[]
    for session in range(2):
        filename = os.path.join(data_dir, "P" + str(sid), "processed", "dataset","session" + str(session + 1) + "_pairs.npy")
        #np.save(filename, np.array(pairs, dtype=object), allow_pickle=True)
        pairs = np.load(filename, allow_pickle=True)
        for pair in pairs:
            #highgamma
            tmp = scaler.fit_transform(np.float32(pair[2][:, pre_EEG:-post_EEG]).transpose())
            z.append(tmp.transpose())  # (118, 1080)
            # EEG
            tmp = scaler.fit_transform(np.float32(pair[1][:,pre_EEG:-post_EEG]).transpose())
            x.append(tmp.transpose()) # (118, 1080)
            # audio
            tmp=pair[0]
            tmp = np.clip(tmp / np.max(np.abs(tmp)),-1,1)
            mel = mel_transformer.mel_spectrogram(torch.from_numpy(tmp)[None,:].to(torch.float32)).numpy().squeeze()
            if window_size is None:
                append_SOS = np.ones((mel_size, 1)) * SOS
                y.append(torch.cat((append_SOS,mel),1).numpy().squeeze())
            else:
                y.append(mel.squeeze())

    # data split
    #raw
    train_x = x[:-10]
    val_x = x[-10:-5]
    test_x = x[-5:]
    #high gamma
    train_x_hg = z[:-10]
    val_x_hg = z[-10:-5]
    test_x_hg = z[-5:]
    #mel
    train_y = y[:-10]
    val_y = y[-10:-5]
    test_y = y[-5:]

    if highgamma:
        train_x=train_x_hg
        val_x=val_x_hg
        test_x=test_x_hg
    if window_size is not None:
        wind_x=window_size
        stride_x=stride
        wind_y=int(window_size/length_ration)
        stride_y=int(stride_x/length_ration)
        train_x2,val_x2,test_x2,train_y2,val_y2,test_y2=[],[],[],[],[],[]
        for x,y in zip(train_x,train_y): # x:(118, 1080),y:(80, 204)
            if x.shape[1] > wind_x+2*sf_EEG*extra_EEG_pairring+pre_EEG+post_EEG:
                tmp1,tmp2=fold_2d23d2(x,y,wind_x,wind_y,stride_x,stride_y,pre_EEG,post_EEG)
                train_x2.append(np.asarray(tmp1))
                a=np.asarray(tmp2)
                append_SOS = np.ones((a.shape[0],a.shape[1], 1)) * SOS
                train_y2.append(np.concatenate((append_SOS,a),axis=2))
            else:
                print("wind_size is larger than speech duration.")
        for x,y in zip(val_x,val_y): # x:(118, 1080),y:(80, 204)
            if x.shape[1] > wind_x+2*sf_EEG*extra_EEG_pairring+pre_EEG+post_EEG:
                tmp1,tmp2=fold_2d23d2(x,y,wind_x,wind_y,stride_x,stride_y,pre_EEG,post_EEG)
                val_x2.append(np.asarray(tmp1))
                a = np.asarray(tmp2)
                append_SOS = np.ones((a.shape[0], a.shape[1], 1)) * SOS
                val_y2.append(np.concatenate((append_SOS, a), axis=2))
        for x,y in zip(test_x,test_y): # x:(118, 1080),y:(80, 204)
            if x.shape[1] > wind_x+2*sf_EEG*extra_EEG_pairring+pre_EEG+post_EEG:
                tmp1,tmp2=fold_2d23d2(x,y,wind_x,wind_y,stride_x,stride_y,pre_EEG,post_EEG)
                test_x2.append(np.asarray(tmp1))
                a = np.asarray(tmp2)
                append_SOS = np.ones((a.shape[0], a.shape[1], 1)) * SOS
                test_y2.append(np.concatenate((append_SOS, a), axis=2))

        train_x2 = np.concatenate(train_x2,axis=0) # (1596 samples, 118channel, 400time)
        val_x2 = np.concatenate(val_x2, axis=0)
        test_x2 = np.concatenate(test_x2, axis=0)

        train_y2 = np.concatenate(train_y2, axis=0)
        val_y2 = np.concatenate(val_y2, axis=0)
        test_y2 = np.concatenate(test_y2, axis=0)

        train_ds = myDataset(train_x2, train_y2)
        val_ds = myDataset(val_x2, val_y2)
        test_ds = myDataset(test_x2, test_y2)

        return train_ds, val_ds, test_ds

    train_ds = myDataset(train_x, train_y)
    val_ds = myDataset(val_x, val_y)
    test_ds = myDataset(test_x, test_y)

    return train_ds,val_ds,test_ds

def dataset_features(sid,wind_size,stride,pre_EEG,post_EEG):
    for session in range(2):
        filename = os.path.join(data_dir, "P" + str(sid), "processed", "dataset","session" + str(session + 1) + "_EEG_features.npy")
        #np.save(filename, np.array(pairs, dtype=object), allow_pickle=True)
        EEG = np.load(filename, allow_pickle=True) # (150, 48826)
        filename = os.path.join(data_dir, "P" + str(sid), "processed", "dataset","session" + str(session + 1) + "_mel_features.npy")
        # np.save(filename, np.array(pairs, dtype=object), allow_pickle=True)
        mel = np.load(filename, allow_pickle=True)  # (40, 48826)





