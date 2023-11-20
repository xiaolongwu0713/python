from example_transformer.EEG_Transformer_seq2seq_master.lib.train import *
# if error happens, change this: "from scipy.misc import logsumexp" to "from scipy.special import logsumexp"
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
npr.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)

from example_speech.closed_loop_seeg_speech_synthesis_master_bak.test import get_data
from speech_Ruijin.utils import fold_2d23d

from speech_Ruijin.config import data_dir
from datetime import datetime
import pytz

the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
result_dir=data_dir+'seq2seq_transformer/'
result_dir=result_dir + the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'/'
print('Result dir: '+ result_dir+'.')


dataname='SingleWordProductionDutch' # 'SingleWordProductionDutch' (10 subjects)/'stereoEEG2speech_master'(3 subjects)
sid=3
x, y = get_data(dataname=dataname, sid=sid,continous_data=True)  # x: (512482,127), y:(344858, 80)
xy_ratio = y.shape[0] / x.shape[0]

lenx=x.shape[0]
leny=y.shape[0]
train_x=x[:int(lenx * 0.8),:]
val_x=x[int(lenx * 0.8):int(lenx * 0.9),:]
test_x=x[int(lenx * 0.9):,:]
train_y=y[:int(leny * 0.8),:]
val_y=y[int(leny * 0.8):int(leny * 0.9),:]
test_y=y[int(leny * 0.9):,:]

win=100
shift=10
win_x, win_y, shift_x, shift_y = win, win* xy_ratio, shift, shift * xy_ratio
x_test,y_test=fold_2d23d(test_x.transpose(),test_y.transpose(), win_x, win_y, shift_x,shift_y)
x_test,y_test=x_test.transpose(0,2,1),y_test.transpose(0,2,1)
# Get input_d, output_d, timesteps from the initial dataset
input_d, output_d = x_test.shape[2], y_test.shape[2]
out_len = y_test.shape[1]
print('input_d:',input_d,'output_d:',output_d,',output length:',out_len)

from common_dl import myDataset
dataset_test = myDataset(x_test, y_test)
batch_size = 64
dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False)

opt = {}
opt['Transformer-layers'] = 6 # 6
opt['Model-dimensions'] = 256 # 256
opt['feedford-size'] = 512 # 512
opt['headers'] = 8
opt['dropout'] = 0.2
opt['src_d'] = input_d # input dimension
opt['tgt_d'] = output_d # output dimension
opt['out_len'] = out_len

model = make_model(opt['src_d'],opt['tgt_d'],opt['Transformer-layers'],
                   opt['Model-dimensions'],opt['feedford-size'],opt['headers'],opt['dropout']).to(device)

from common_dl import device
#device=torch.device('cpu')
fold_date='2023_07_09_23_05'
testing_epoch=18
path_file='H:/Long/data/speech_Ruijin/seq2seq_transformer/'+fold_date+'/'+str(testing_epoch)+'.pth'
#path_file='/Users/xiaowu/tmp/models/'+str(testing_epoch)+'.pth'

checkpoint = torch.load(path_file,map_location=device)
model=model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
#fig,ax=plt.subplots(2,1, figsize=(10,20))

# loop through many test data
encoder_only=False
predictions=[]
truths=[]
model.eval()
for i,(test_x, test_y) in tqdm(enumerate(dataloader_test)):
    # batch inference
    pred, tgt = output_prediction2(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1, output_d=opt['tgt_d'],
                                   device=device, encoder_only=encoder_only)

    ''' individual inference
    test_x, test_y = next(iter(dataloader_test))
    test_x, test_y = test_x.numpy(), test_y.numpy()
    test_x, test_y = test_x[0], test_y[0]
    pred, tgt = output_prediction(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1, output_d=opt['tgt_d'],
                                   device=device,encoder_only=encoder_only)
    '''
    predictions.append(pred.numpy())
    truths.append(test_y.numpy())

    #ax[0].imshow(tgt[-1, :, :].squeeze(), cmap='RdBu_r')
    #ax[1].imshow(pred[-1, :, :].squeeze(), cmap='RdBu_r')
    #filename = path_file[:-4]+'/'+str(i)+'.png'
    #fig.savefig(filename)

    #if i>400:
    #    break
predictions2=np.concatenate(predictions,axis=0)
truths2=np.concatenate(truths,axis=0)
filename=result_dir+'predictions.npy'
np.save(filename,predictions2)
filename=result_dir+'truths.npy'
np.save(filename,truths2)

'''
## manual check the testing result
test_x,test_y=next(iter(dataloader_test)) # torch.Size([1, 1000, 127])
test_x,test_y=test_x[0].numpy(),test_y[0].numpy()
pred, truth = output_prediction(model,test_x, test_y, max_len=opt['out_len'], start_symbol=1,output_d=opt['tgt_d'])

ax[0].imshow(truth[-1, :, :].squeeze(), cmap='RdBu_r')
ax[1].imshow(pred[-1, :, :].squeeze(), cmap='RdBu_r')

idx=0
filename='H:/Long/data/speech_Ruijin/seq2seq_transformer/2023_07_08_09_43/'+str(idx)+'_pred.npy'
np.save(filename,pred.squeeze())
filename='H:/Long/data/speech_Ruijin/seq2seq_transformer/2023_07_08_09_43/'+str(idx)+'_truth.npy'
np.save(filename,truth.squeeze())

'''

import numpy as np
win=100
shift=10
filename='/Users/xiaowu/My Drive/tmp/prediction/predictions.npy'
pred=np.load(filename) # (3066, 100, 40)
filename='/Users/xiaowu/My Drive/tmp/prediction/truths.npy'
tgt=np.load(filename) # (3066, 100, 40)
# step 0 are all 1s
pred[:,0,:]=pred[:,1,:]
tgt[:,0,:]=tgt[:,1,:]
def averaging(pred,win,shift):
    h=pred.shape[0] # 3066
    d=pred.shape[2] # 40
    w=win+(h-1)*shift # 30750
    summary=np.zeros((w,d)) # (30750, 40)
    for i in range(len(pred)):
        start=i*shift
        end=start+win
        summary[start:end,:]=summary[start:end,:]+pred[i]

    stairs=np.ones(win)

    subwin=10
    sublist=[1,]*subwin
    steps=int(win/subwin)
    for i in range(steps):
        start=i*subwin
        stop=start+subwin
        stairs[start:stop]=[j*(i+1) for j in sublist]

    stairs_reverse=[stairs[len(stairs)-i-1] for i in range(len(stairs))]
    middle=[10,]*(summary.shape[0]-len(stairs)-len(stairs_reverse))
    repeat=list(stairs)+middle+stairs_reverse # 30750

    average=np.zeros(summary.shape) # (30750, 40)
    for i in range(summary.shape[0]):
        average[i,:]=summary[i,:]/repeat[i]

    return average

avg_pred=averaging(pred,win,shift)
avg_tgt=averaging(tgt,win,shift)

fig,ax=plt.subplots(2,1)
start=0
end=30740
ax[0].imshow(avg_tgt[start:end,:].transpose(),cmap='RdBu_r', aspect='auto')
ax[1].imshow(avg_pred[start:end,:].transpose(),cmap='RdBu_r', aspect='auto')