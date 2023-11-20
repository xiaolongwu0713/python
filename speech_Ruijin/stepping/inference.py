import io
import sys
import socket
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from gesture.models.d2l_resnet import d2lresnet
from speech_Ruijin.stepping.dataset import dataset_stepping
from speech_Ruijin.stepping.model_stepping import tsception

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
from datetime import datetime
import pytz
from speech_Ruijin.config import *
from common_dl import *
from torch.utils.data import DataLoader
set_random_seeds(999)


dataname='SingleWordProductionDutch'
sid=3
if dataname=='SingleWordProductionDutch':
    pt_folder='sub-' + f"{sid:02d}"
    from speech_Ruijin.stepping.opt import opt_SingleWordProductionDutch as opt
else:
    pt_folder='sid'+str(sid)
    from speech_Ruijin.stepping.opt import opt_mydata as opt

mel_bins=opt['mel_bins']
sf_EEG=opt['sf_EEG']
batch_size=opt['batch_size']
wind_size = opt['wind_size']
stride = opt['stride']
epoch_num = opt['epoch_num']
patients = opt['patients']
learning_rate=opt['learning_rate'] #0.005
dropout=opt['dropout']
sids=[1,2,3,4,5,6,7,8,9,10]

folder_dates=['2023_08_12_16_46','2023_08_12_17_56','2023_08_12_19_13','2023_08_12_20_20','2023_08_12_21_39',
                  '999','999','999','999','999']
epochs=[2,4,1,5,8,
        999,999,999,999,999]

sid_idx = [1,2,3,4,5]  # test on the sid_idx_th sids
for sid,folder_date, epoch in zip([sids[i-1] for i in sid_idx],
                        [folder_dates[i-1] for i in sid_idx],[epochs[i-1] for i in sid_idx]):
    print('sid: '+str(sid)+'.')
    shift=None
    test_ds = dataset_stepping(dataname=dataname,sid=sid,wind_size=wind_size,
                                                 stride=stride, mel_bins=mel_bins,shift=shift,test_only=True)

    test_size=len(test_ds)
    print('test size: '+str(test_size)+'.')
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    ch_num=test_ds[0][0].shape[0]
    learning_rate = opt['learning_rate']  # 0.005
    dropout = opt['dropout']
    net_name = opt['net_name']
    if net_name == 'tsception':
        net = tsception(1000, ch_num, wind_size, 6, 6, dropout, mel_bins)
    elif net_name == 'resnet':
        net = d2lresnet(task='regression', reg_d=mel_bins)
    net = net.to(device)

    from pre_all import device
    if computer == 'workstation':
        result_dir = data_dir + 'stepping/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + \
                     str(sid) + '_wind_'+str(wind_size)+ '/' + folder_date + '/'
    elif computer == 'mac':
        result_dir = '/Users/xiaowu/tmp/models/stepping/' + folder_date + '/'
    path_file = result_dir + 'best_model_epoch' + str(epoch) + '.pth'  # '19_final.pth'
    checkpoint = torch.load(path_file,map_location=device)
    net.load_state_dict(checkpoint['net'])
    loss_fun = nn.MSELoss()

    net.eval()
    truths=[]
    preds=[]
    with torch.no_grad():
        for batch, (x, y) in tqdm(enumerate(test_loader)):
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = net(x)  # torch.Size([200, 23])
            loss = loss_fun(y_pred, y)  # y: torch.Size([3, 80])
            truths.append(y.cpu().numpy().squeeze())
            preds.append(y_pred.cpu().numpy().squeeze())
    truths=np.concatenate(truths,axis=0)
    preds=np.concatenate(preds,axis=0)
    # flatten list of 2D to 2D
    corr = []  # transformer: mean=0.8677304214419497 seq2seq:0.8434792922232249
    #test_on_shorter_range=3000
    test_on_shorter_range=truths.shape[0]
    for specBin in range(truths.shape[1]):
        r, p = pearsonr(preds[:test_on_shorter_range, specBin], truths[:test_on_shorter_range, specBin])
        corr.append(r)
    mean_corr=sum(corr) / len(corr)
    print(mean_corr)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(preds, truths)
    print('MSE:' + str(mse))

    fig,ax=plt.subplots(2,1, figsize=(6,4))
    ax[0].imshow(preds.transpose(),aspect='auto')
    ax[1].imshow(truths.transpose(),aspect='auto')
    fig.tight_layout()
    figname = result_dir + 'plot.png'
    fig.savefig(figname)

    filename = result_dir + 'result.txt'
    with open(filename, 'w') as f:
        f.write(str(mean_corr))
        f.write('\n')
        f.write(str(mse))



