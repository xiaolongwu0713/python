from example_transformer.EEG_Transformer_seq2seq_master.lib.train import *
# if error happens, change this: "from scipy.misc import logsumexp" to "from scipy.special import logsumexp"
import PIL
import io
from torchvision.transforms import ToTensor
import numpy as np
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
npr.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)

from example_speech.closed_loop_seeg_speech_synthesis_master_bak.test import get_data
from speech_Dutch.utils import fold_2d23d

from speech_Dutch.config import data_dir
from datetime import datetime
import pytz
from tensorboardX import SummaryWriter

testing=False
print('Testing mode:'+str(testing))
the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
result_dir=data_dir+'seq2seq_transformer/'
result_dir=result_dir + the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'/'
print('Result dir: '+ result_dir+'.')
writer = SummaryWriter(result_dir)

continous_data=True
dataname='SingleWordProductionDutch' # 'SingleWordProductionDutch' (10 subjects)/'stereoEEG2speech_master'(3 subjects)
x, y = get_data(dataname=dataname, sid=3,continous_data=continous_data)  # x: (512482,127), y:(344858, 80)
xy_ratio = y.shape[0] / x.shape[0]

norm_mel=False
if norm_mel:
    mu = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    y = (y - mu) / std

lenx=x.shape[0]
leny=y.shape[0]
train_x=x[:int(lenx * 0.8),:] # (246017, 127)
val_x=x[int(lenx * 0.8):int(lenx * 0.9),:]
test_x=x[int(lenx * 0.9):,:]
train_y=y[:int(leny * 0.8),:]
val_y=y[int(leny * 0.8):int(leny * 0.9),:]
test_y=y[int(leny * 0.9):,:]

norm_EEG=False
if norm_EEG:
    mu=np.mean(train_x,axis=0)
    std=np.std(train_x,axis=0)
    train_x=(train_x-mu)/std
    val_x=(val_x-mu)/std
    test_x=(test_x-mu)/std

# shape: (1000batch, 300time, 10channel)
win=100
if testing:
    shift=100
else:
    shift=2
win_x, win_y, shift_x, shift_y = win, win* xy_ratio, shift, shift * xy_ratio
x_train,y_train=fold_2d23d(train_x.transpose(),train_y.transpose(), win_x, win_y, shift_x,shift_y)
x_val,y_val=fold_2d23d(val_x.transpose(),val_y.transpose(), win_x, win_y, shift_x,shift_y)
x_train,y_train,x_val,y_val=[x.transpose(0,2,1) for x in (x_train,y_train,x_val,y_val)]

win_x, win_y, shift_x, shift_y = 100, 100* xy_ratio, 100, 100 * xy_ratio
x_test,y_test=fold_2d23d(test_x.transpose(),test_y.transpose(), win_x, win_y, shift_x,shift_y)
x_test,y_test=x_test.transpose(0,2,1),y_test.transpose(0,2,1)

# Get input_d, output_d, timesteps from the initial dataset
input_d, output_d = x_test.shape[2], y_test.shape[2]
out_len = y_test.shape[1]
print('input_d:',input_d,'output_d:',output_d,',output length:',out_len)

from common_dl import myDataset
dataset_train = myDataset(x_train, y_train)
dataset_val = myDataset(x_val, y_val)
dataset_test = myDataset(x_test, y_test)

batch_size = 64
dataloader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
dataloader_val = DataLoader(dataset=dataset_val,batch_size=batch_size,shuffle=True)
#batch_size=1
dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False)

'''
class LDSDataset(Dataset):
    # use boolen value to indicate that the data is for training or testing
    def __init__(self,x,y,train=True,ratio=0.7):
        self.len = x.shape[0]
        self.ratio = ratio
        split = int(self.len*self.ratio)
        split1 = int(self.len * 0.7)
        split2 = int(self.len * 0.9)
        self.x_train = torch.from_numpy(x[:split1])
        self.y_train = torch.from_numpy(y[:split1])
        self.x_val = torch.from_numpy(x[split1:split2])
        self.y_val = torch.from_numpy(y[split1:split2])
        self.x_test = torch.from_numpy(x[split2:])
        self.y_test = torch.from_numpy(y[split2:])
        self.train = train

    def __len__(self):
        if self.train:
            return int(self.len*self.ratio)
        else:
            return int(self.len*(1-self.ratio))

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]

# split training and testing set
split_ratio = 0.7
batch_size = 50
dataset_train = LDSDataset(ts,ls,train=True,ratio=split_ratio)
dataloader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
dataset_test = LDSDataset(ts,ls,False,split_ratio)
dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=True)
'''

opt = {}
opt['norm_mel'] = norm_mel
opt['Transformer-layers'] = 2 # 6
opt['Model-dimensions'] = 32# 256
opt['feedford-size'] = 64 # 512
opt['headers'] = 2
opt['dropout'] = 0.5
opt['src_d'] = input_d # input dimension
opt['tgt_d'] = output_d # output dimension
opt['out_len'] = out_len

from pre_all import device
criterion = nn.MSELoss() # mean squared error
# setup model using hyperparameters defined above
encoder_only=False
model = make_model(opt['src_d'],opt['tgt_d'],opt['Transformer-layers'],
                   opt['Model-dimensions'],opt['feedford-size'],opt['headers'],opt['dropout'],opt['norm_mel'],encoder_only=encoder_only).to(device)

# setup optimization function
model_opt = NoamOpt(model_size=opt['Model-dimensions'], factor=1, warmup=400,
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9))
optimizer = torch.optim.Adam(model.parameters(), lr=0.015, betas=(0.9, 0.98), eps=1e-9)

resume=False
if resume:
    print('Load pre-trained model.')
    check_file='H:/Long/data/speech_Ruijin/seq2seq_transformer/2023_07_07_17_04/99.pth'
    from common_dl import device
    checkpoint = torch.load(check_file,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch=checkpoint['epoch']
else:
    resume_epoch=0

fig,ax=plt.subplots(2,1, figsize=(10,20))
total_epoch = 200
train_losses = []
val_losses = []
patients=20
from tqdm import tqdm
for epoch in tqdm(range(total_epoch)):
    epoch = epoch + resume_epoch
    model.train()
    train_loss,pred,truth = run_epoch(data_gen(dataloader_train,encoder_only=encoder_only), model,SimpleLossCompute(model.generator, criterion, model_opt))
    #train_loss, pred, truth = run_epoch(data_gen(dataloader_train), model,criterion,optimizer)
    train_losses.append(train_loss)

    ax[0].imshow(truth.cpu()[-1,:, :].numpy().squeeze(), cmap='RdBu_r')
    ax[1].imshow(pred.cpu()[-1,:, :].detach().numpy().squeeze(), cmap='RdBu_r')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('Image/train', image[0], epoch)
    ax[0].clear()
    ax[1].clear()

    print('Validating...')
    model.eval() # test the model
    predictions = []
    truths = []
    for i, (test_x, test_y) in enumerate(dataloader_val):
        # batch inference
        pred, tgt = output_prediction2(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1,
                                       output_d=opt['tgt_d'],
                                       device=device, encoder_only=encoder_only)
        predictions.append(pred)
        truths.append(test_y)
    val_loss = criterion(torch.cat(predictions).contiguous(), torch.cat(truths).contiguous())
    ''' This is a bad validating method, because it uses teacher force like in the training step.
    with torch.no_grad():
        val_loss,pred,truth = run_epoch(data_gen(dataloader_val,encoder_only=encoder_only), model,SimpleLossCompute(model.generator, criterion, None))
        #val_loss, pred, truth = run_epoch(data_gen(dataloader_val), model,criterion, None)
        val_losses.append(val_loss)

    ax[0].imshow(truth.cpu()[-1, :, :].numpy().squeeze(), cmap='RdBu_r')
    ax[1].imshow(pred.cpu()[-1, :, :].detach().numpy().squeeze(), cmap='RdBu_r')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('Image/val', image[0], epoch)
    ax[0].clear()
    ax[1].clear()
    '''

    if epoch==resume_epoch:
        best_val_loss=val_loss
        patient=patients
        best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }
    else:
        if val_loss>best_val_loss:
            patient=patient-1
        else:
            best_val_loss = val_loss
            patient=patients
            best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }
    print('Epoch[{}/{}], train_loss: {:.6f},val_loss: {:.6f}. Patient: {:d}'.format(epoch + 1, total_epoch+resume_epoch, train_loss, val_loss,patient))

    ## plot testing result every epoch (because training is too slow)
    # not suitable for encoder_only mode; just use the validation dataset to monitor the model performance \
    # (no teacher force in encoder_only mode)
    print('Testing...')
    for i, (test_x, test_y) in enumerate(dataloader_test):
        #if i > 5:
        test_x, test_y = test_x.numpy(), test_y.numpy()
        test_x, test_y = test_x[0], test_y[0]
        # make a prediction then compare it with its true output
        pred, truth = output_prediction(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1,
                                        output_d=opt['tgt_d'], device=device,encoder_only=encoder_only)

        ax[0].imshow(truth[-1, :, :].squeeze(), cmap='RdBu_r')
        ax[1].imshow(pred[-1, :, :].squeeze(), cmap='RdBu_r')
        filename = result_dir + str(epoch) + '_' + str(i) + '.png'
        fig.savefig(filename)

        if i == 20:
            break
    if patient==0:
        #pass # validation using teacher force can't reflect the true generalization
        break

    if testing:
        break
filename=result_dir+str(best_model['epoch'])+'.pth'
torch.save(best_model,filename)

final_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }

filename=result_dir+str(epoch)+'_final.pth'
torch.save(final_model,filename)


from common_dl import device
#device=torch.device('cpu')
checkpoint = torch.load(filename,map_location=device)
model=model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])


'''
fig,ax=plt.subplots(2,1, figsize=(10,20))
for i,(test_x, test_y) in enumerate(dataloader_test):
    if i>5:
        test_x, test_y=test_x.numpy(), test_y.numpy()
        test_x, test_y=test_x[0], test_y[0]
        # make a prediction then compare it with its true output
        pred, truth = output_prediction(model,test_x, test_y, max_len=opt['out_len'], start_symbol=1,output_d=opt['tgt_d'],device=device)
    
        ax[0].imshow(truth[-1, :, :].squeeze(), cmap='RdBu_r')
        ax[1].imshow(pred[-1, :, :].squeeze(), cmap='RdBu_r')
        filename = result_dir+str(epoch)+'_' + str(i) + '.png'
        fig.savefig(filename)
    
        if i==10:
            break

'''

