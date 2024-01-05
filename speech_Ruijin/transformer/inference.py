'''
The inference script for both seq2seq and the transformer models.
'''

import sys,os,glob
import sys
import socket

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['C:/Users/xiaowu/mydrive/python/'])


from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pre_all import computer, device, running_from_CMD
from common_dl import myDataset
from speech_Ruijin.baseline_linear_regression.reconstruction_minimal import createAudio
from speech_Ruijin.config import data_dir
import numpy as np
import numpy.random as npr
from speech_Ruijin.transformer.lib.train import make_model #*
# if error happens, change this: "from scipy.misc import logsumexp" to "from scipy.special import logsumexp"
import torch
from torch.utils.data import DataLoader
from speech_Ruijin.transformer.utils import test_transformer_model, test_seq2seq_model, averaging
from speech_Ruijin.transformer.opt import opt_transformer
import math
npr.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)

from speech_Ruijin.baseline_linear_regression.extract_features import dataset
from speech_Ruijin.utils import fold_2d23d

if running_from_CMD:
    sid = int(float(sys.argv[1]))
    dataname = sys.argv[2] #'SingleWordProductionDutch'
    time_stamp=sys.argv[3]
else:
    dataname='SingleWordProductionDutch'
    sid=3
    time_stamp='dummy'
if dataname=='mydata':#'SingleWordProductionDutch'/'mydata' # 'SingleWordProductionDutch' (10 subjects)/'stereoEEG2speech_master'(3 subjects)
    from speech_Ruijin.transformer.opt import opt_mydata as opt
    test_shift = opt['test_shift']  # 300
# loop through all sids
fig, ax = plt.subplots(2, 1)
import torch.nn as nn
sids=[1,2,3,4,5,6,7,8,9,10]

model_name='seq2seq_transformer' # 'seq2seq_transformer'/'seq2seq'
if model_name=='seq2seq_transformer':
    from speech_Ruijin.transformer.opt import opt_SingleWordProductionDutch as opt

    ############
    sf_EEG = opt['sf_EEG']
    mel_bins = opt['mel_bins']
    stride = opt['stride']
    step_size = opt['step_size']
    model_order = opt['model_order']
    use_the_official_tactron_with_waveglow = opt['use_the_official_tactron_with_waveglow']
    if use_the_official_tactron_with_waveglow:
        target_SR = 22050
        frameshift = 256 / target_SR  # 256 steps in 0.011609977324263039 s
        winL = 1024 / target_SR
    else:
        winL = opt['winL']
        frameshift = opt['frameshift']
    win = math.ceil(opt['win'] / frameshift)  # int: steps
    history = math.ceil(opt['history'] / frameshift)  # int(opt['history']*sf_EEG) # int: steps
    baseline_method = opt['baseline_method']
    stride_test = opt['stride_test']
    ##################

    '''
    folder_dates = ['2023_08_03_16_06', '2023_07_25_18_48', '2023_07_25_18_52', '2023_08_03_11_52', '2023_07_25_19_02',
                    '2023_07_25_19_06', '2023_07_25_19_14', '2023_07_25_19_19', '2023_07_25_19_24', '2023_07_25_19_29']
    epochs = [8, 0, 4, 0, 2,
              14, 3, 4, 3, 1]
    '''

    # 80 mel bins
    folder_dates=['2023_08_31_10_28_14','2023_08_31_10_28_14','2023_08_31_08_41','2023_08_31_10_28_14','2023_08_31_10_28_14',
                  '2023_08_31_10_28_14','2023_08_31_10_28_14','2023_08_31_10_28_14','2023_08_31_10_28_14','2023_08_31_10_28_14']
    #epochs=[3,0,14,5,2,
    #       14,3,4,3,1]


    '''
        use baseline method (mel calculated with Tactron),23 melbins,
            folder_dates=['2023_07_25_21_03','2023_07_25_21_08','2023_07_25_21_12','2023_07_25_21_22','2023_07_25_21_31',
                          '2023_07_25_21_37','2023_07_25_21_45','2023_07_25_21_50','2023_07_25_21_55','2023_07_25_22_01']
            epochs=[2,0,21,17,8,
                    14,1,4,7,9]
        '''
    sid_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # or test on all sids
    #sid_idx=[3]
elif model_name=='seq2seq':
    from speech_Ruijin.seq2seq.opt import opt_SingleWordProductionDutch as opt

    ###############
    mel_bins = opt['mel_bins']
    winL = opt['winL']
    frameshift = opt['frameshift']
    win = math.ceil(opt['win'] / frameshift)  # int: steps
    history = math.ceil(opt['history'] / frameshift)  # int(opt['history']*sf_EEG) # int: steps
    stride = opt['stride']
    stride_test = opt['stride_test']
    baseline_method = opt['baseline_method']
    norm_EEG = opt['norm_EEG']  # True
    norm_mel = opt['norm_mel']  # False
    sf_EEG = opt['sf_EEG']

    embed_size = opt['embed_size']  # = 256
    num_hiddens = opt['num_hiddens']  # = 256
    num_layers = opt['num_layers']  # = 2
    dropout = opt['dropout']  # = 0.5
    lr = opt['lr']
    batch_size = opt['batch_size']
    ##################

    folder_dates = ['2023_08_12_15_30','2023_08_12_15_35', '2023_08_12_15_39','2023_08_12_15_43','2023_08_12_15_47',
                    '2023_08_12_15_50','2023_08_12_15_54','2023_08_12_16_00','2023_08_12_16_03','2023_08_12_16_08',]
    epochs = [4,0, 1,1,0,
              1,7,0,3,3]
    sid_idx = [1,2,3,4,5,6,7,8,9,10]  # test on the sid_idx_th sids

print('baseline_method: ' + str(opt['baseline_method']) +'; win: ' + str(win) + '; history:' +
      str(history) + '; stride:' + str(stride)+'; winL:'+str(winL)+'; frameshift: '+str(frameshift)+'.')

#for sid,folder_date in zip([sids[i-1] for i in sid_idx],[folder_dates[i-1] for i in sid_idx]):
#for sid,folder_date, epoch in zip([sids[i-1] for i in sid_idx],[folder_dates[i-1] for i in sid_idx], [epochs[i-1] for i in sid_idx]):
for aaa in [1]: # no loop anymore
    sid, folder_date = sid, time_stamp
    print('sid: '+str(sid)+'.')
    if dataname == 'mydata':
        test_shift = 300
        from speech_Ruijin.dataset import dataset_Dutch

        # x1, y1 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=1,test_shift=test_shift, mel_bins=mel_bins, continous=continous_data)
        # x2, y2 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=2,test_shift=300, mel_bins=mel_bins, continous=continous_data)
        x1, y1 = dataset(dataset_name=dataname, sid=sid, session=1, test_shift=0, melbins=mel_bins, stacking=False,
                         winL=winL, frameshift=frameshift)
        x2, y2 = dataset(dataset_name=dataname, sid=sid, session=2, test_shift=0, melbins=mel_bins, stacking=False,
                         winL=winL, frameshift=frameshift)
        if False:
            mu = np.mean(x1, axis=0)
            std = np.std(x1, axis=0)
            x1 = (x1 - mu) / std
            mu = np.mean(x2, axis=0)
            std = np.std(x2, axis=0)
            x2 = (x2 - mu) / std

            x = np.concatenate((x1, x2), axis=0)
            y = np.concatenate((y1, y2), axis=0)
        else:
            x = x1
            y = y1
    elif dataname == 'SingleWordProductionDutch':
        # x, y = get_data(dataname=dataname, sid=sid,continous_data=continous_data,mel_bins=mel_bins)  # x: (512482,127), y:(344858, 80)
        x, y = dataset(dataset_name=dataname, sid=sid, melbins=mel_bins, stacking=False, winL=winL,  target_SR =target_SR,frameshift=frameshift
                       ,use_the_official_tactron_with_waveglow=use_the_official_tactron_with_waveglow)

    xy_ratio = y.shape[0] / x.shape[0]

    if opt['norm_mel']:
        mu = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        y = (y - mu) / std

    lenx = x.shape[0]
    leny = y.shape[0]
    train_x = x[:int(lenx * 0.8), :]
    val_x = x[int(lenx * 0.8):int(lenx * 0.9), :]
    test_x = x[int(lenx * 0.9):, :]
    train_y = y[:int(leny * 0.8), :]
    val_y = y[int(leny * 0.8):int(leny * 0.9), :]
    test_y = y[int(leny * 0.9):, :] #(3003, 80)

    #norm_EEG = opt['norm_EEG'] #False
    if opt['norm_EEG']:
        mu = np.mean(train_x, axis=0)
        std = np.std(train_x, axis=0)
        train_x = (train_x - mu) / std
        val_x = (val_x - mu) / std
        test_x = (test_x - mu) / std

    win_x, win_y, shift_x, shift_y = win + history, int(win * xy_ratio), stride_test, int(stride_test * xy_ratio)
    # win_x, win_y, shift_x, shift_y = win, win* xy_ratio, stride, stride * xy_ratio
    x_test, y_test = fold_2d23d(test_x.transpose(), test_y[history:,:].transpose(), win_x, win_y, shift_x, shift_y)
    x_test, y_test = x_test.transpose(0,2,1), y_test.transpose(0,2,1) #0, 2, 1
    # Get input_d, output_d, timesteps from the initial dataset
    input_d, output_d = x_test.shape[2], y_test.shape[2]
    out_len = y_test.shape[1] #(batch,feature,time)-->(batch,time, feature)
    # encoder need: (time, batch, feature)
    dataset_test = myDataset(x_test, y_test)
    batch_size = opt_transformer['batch_size']
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    if model_name=='seq2seq_transformer':
        # device=torch.device('cpu')
        if computer == 'workstation' or computer == 'Yoga':
            result_dir = data_dir + 'seq2seq_transformer/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(
                sid) + '/' + folder_date + '/'
        elif computer == 'mac':
            result_dir = '/Users/xiaowu/tmp/models/seq2seq_transformer/' + folder_date + '/'

        path_file = result_dir +  'best_model_epoch*'+'.pth' # '19_final.pth'
        path_file = os.path.normpath(glob.glob(path_file)[0])

        opt_transformer['src_d'] = input_d  # input dimension
        opt_transformer['tgt_d'] = output_d  # output dimension
        opt_transformer['out_len'] = out_len

        encoder_only = False
        spatial_attention = False
        model = make_model(opt_transformer['src_d'], opt_transformer['tgt_d'], N=opt_transformer['Transformer-layers'],
                           d_model=opt_transformer['Model-dimensions'], d_ff=opt_transformer['feedford-size'], h=opt_transformer['headers'],
                           dropout=opt_transformer['dropout'], norm_mel=opt['norm_mel'], encoder_only=encoder_only,
                           spatial_attention=spatial_attention).to(device)
        model = model.to(device)
        checkpoint = torch.load(path_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        pred, tgt = test_transformer_model(sid, model, opt_transformer, dataloader_test, result_dir)
        filename=result_dir+'/prediction.npy'
        np.save(filename,pred)
        filename = result_dir + '/truth.npy'
        np.save(filename, tgt)
        # or load prediction
        '''
        the_date = '2023_07_13_17_26'
        if computer == 'mac':
            filename_pred = '/Users/xiaowu/tmp/untitled folder/prediction.npy'
            filename_tgt = '/Users/xiaowu/tmp/untitled folder/truth.npy'
        elif computer == 'workstation':
            filename_pred = 'H:/Long/data/speech_Ruijin/seq2seq_transformer/' + the_date + '/best_model_epoch53_predictions.npy'
            filename_tgt = 'H:/Long/data/speech_Ruijin/seq2seq_transformer/' + the_date + '/best_model_epoch53_predictions.npy'
        pred = np.load(filename_pred)  # (3066, 100, 40)
        tgt = np.load(filename_tgt)  # (3066, 100, 40)
        '''

        pred[:, 0, :] = pred[:, 1, :]
        tgt[:, 0, :] = tgt[:, 1, :]

        avg_tgt = averaging(tgt, win_y, shift_y)  # (30750, 40)
        avg_pred = averaging(pred, win_y, shift_y)  # (6121, 100, 40)-->(6220, 40)

    # seq2seq output/prediction has 99 timesteps, so append an extra one step
    if model_name=='seq2seq':
        from speech_Ruijin.seq2seq.model_d2l import Seq2SeqEncoder2, Seq2SeqAttentionDecoder, EncoderDecoder
        if computer == 'workstation':
            result_dir = data_dir + 'seq2seq/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(
                sid) + '/' + folder_date + '/'
        elif computer == 'mac':
            result_dir = '/Users/xiaowu/tmp/models/seq2seq_transformer/' + folder_date + '/'
        path_file = result_dir + 'best_model_epoch' + str(epoch)+'.pth'

        out_features = dataset_test[0][1].shape[1]
        ch_num = dataset_test[0][0].shape[1]
        in_features = ch_num
        embedding = True
        bidirectional = True  # not implemented
        if not embedding:
            num_hiddens = out_features
        encoder = Seq2SeqEncoder2(in_features, embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(out_features, embed_size, num_hiddens, num_layers, dropout=dropout,
                                          batch_size=batch_size)
        model = EncoderDecoder(encoder, decoder).to(device)
        checkpoint = torch.load(path_file, map_location=device)
        model.load_state_dict(checkpoint['net'])
        del checkpoint
        pred, tgt = test_seq2seq_model(model, dataloader_test, result_dir)

        tgt = np.concatenate((tgt[:, 0, :][:, np.newaxis, :], tgt), axis=1)
        pred = np.concatenate((pred[:, 0, :][:, np.newaxis, :], pred), axis=1)

        filename = result_dir + '/prediction.npy'
        np.save(filename, pred)
        filename = result_dir + '/truth.npy'
        np.save(filename, tgt)

        avg_tgt = averaging(tgt, win_y, shift_y)  # (30750, 40)
        avg_pred = averaging(pred, win_y, shift_y)  # (6121, 100, 40)-->(6220, 40)
    del model
    start = 0
    #ax[0].imshow(avg_tgt.transpose(), cmap='RdBu_r', aspect='auto')
    #ax[1].imshow(avg_pred.transpose(), cmap='RdBu_r', aspect='auto')
    avg_tgt=avg_tgt[10:-10,:]
    avg_pred=avg_pred[10:-10,:]
    vmin=avg_tgt.min()
    vmax=avg_tgt.max()
    ax[0].imshow(avg_tgt.transpose(),aspect='auto',vmin=vmin,vmax=vmax)
    ax[1].imshow(avg_pred.transpose(),aspect='auto',vmin=vmin,vmax=vmax)
    figname=result_dir+'plot.png'
    fig.savefig(figname)
    ax[0].clear()
    ax[1].clear()

    filename = result_dir + 'mel_pred.npy'
    np.save(filename, avg_pred)
    filename = result_dir + 'mel_tgt.npy'
    np.save(filename, avg_tgt)
    #avg_tgt=avg_tgt.transpose()
    #avg_pred=avg_pred.transpose()
    partial_teste = True  # only test partial testing data in the original paper
    if partial_teste:
        test_on_shorter_range = 3000  # mean=0.82
    else:
        test_on_shorter_range = avg_pred.shape[0]
    corr = []  # transformer: mean=0.8677304214419497 seq2seq:0.8434792922232249
    for specBin in range(avg_pred.shape[1]):
        r, p = pearsonr(avg_pred[:test_on_shorter_range, specBin], avg_tgt[:test_on_shorter_range, specBin])
        corr.append(r)
    mean_corr=sum(corr) / len(corr)
    print('corr:'+str(mean_corr))

    criterion = nn.MSELoss()
    loss = criterion(torch.from_numpy(avg_pred),torch.from_numpy(avg_tgt))  # 40:10.6792; 80: tensor(18.4728, dtype=torch.float64)
    from sklearn.metrics import mean_squared_error
    loss=mean_squared_error(avg_pred, avg_tgt)
    print('MSE:'+str(loss))

    filename = result_dir + 'result.txt'
    with open(filename, 'w') as f:
        f.write(str(mean_corr))
        f.write('\n')
        f.write(str(loss.item()))

    gen_wave=False
    if gen_wave:
        # Synthesize waveform from spectrogram using Griffin-Lim
        print("Synthesizing waveform using Griffin-Lim...")
        audiosr = 10240
        winLength = 1024 / 10240
        frameshift = 10 / 10240
        import scipy.io.wavfile as wavfile
        avg_tgt2 = createAudio(avg_tgt, audiosr=audiosr, winLength=winLength, frameshift=frameshift, log=True)
        avg_pred2 = createAudio(avg_pred, audiosr=audiosr, winLength=winLength, frameshift=frameshift, log=True)
        wavfile.write(result_dir+'tgt.wav', int(audiosr), avg_tgt2)
        wavfile.write(result_dir+'pred.wav', int(audiosr), avg_pred2)


'''
# Synthesize using Nvidia WaveGlow. ( It needs 80 frequency bins)
import torch
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
#waveglow = waveglow.to('cpu')
waveglow.eval()
with torch.no_grad():
    audio_tgt = waveglow.infer(torch.from_numpy(avg_tgt.transpose())[None,:,:].to(dtype=torch.float).to('cpu'))
    audio_pred = waveglow.infer(torch.from_numpy(avg_pred.transpose())[None,:,:].to(dtype=torch.float).to('cpu'))

audio_numpy_tgt = audio_tgt[0].data.cpu().numpy()
audio_numpy_pred = audio_pred[0].data.cpu().numpy()
rate = 22050 # higher rate will speed up the audio
np.save('C:/Users/wuxiaolong/我的云端硬盘/tmp/model/audio_numpy_tgt.npy',audio_numpy_tgt)
np.save('C:/Users/wuxiaolong/我的云端硬盘/tmp/model/audio_numpy_pred.npy',audio_numpy_pred)

import sounddevice as sd
fs=10240
sd.play(avg_tgt2,fs)
sd.play(avg_pred2,fs)
sd.stop()
'''