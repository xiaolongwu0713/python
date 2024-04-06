import os
import sys
import socket

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])

from speech_Ruijin.baseline_linear_regression.extract_features import dataset
from speech_Ruijin.transformer.lib.train import *
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
from speech_Ruijin.utils import fold_2d23d
from speech_Ruijin.transformer.opt import opt_transformer
from speech_Ruijin.config import data_dir
from datetime import datetime
import pytz
from tensorboardX import SummaryWriter
from pre_all import computer, debugging, running_from_CMD
from pre_all import set_random_seeds
set_random_seeds(99999)
if running_from_CMD:
    sid = int(float(sys.argv[1]))
    dataname = sys.argv[2] #'SingleWordProductionDutch'
    time_stamp=sys.argv[3]
    mel_bins=sys.argv[4]
    testing=False
else:
    sid=3
    dataname = 'SingleWordProductionDutch'
    time_stamp='testing_time'
    mel_bins=80
    testing=False
testing=(testing or debugging or computer=='mac')

if dataname=='mydata':
    from speech_Ruijin.transformer.opt import opt_mydata as opt
    test_shift = opt['test_shift']#300
elif dataname=='SingleWordProductionDutch':
    from speech_Ruijin.transformer.opt import opt_SingleWordProductionDutch as opt


###############
target_SR=opt['target_SR']
use_the_official_tactron_with_waveglow=opt['use_the_official_tactron_with_waveglow']
window_eeg=opt['window_eeg'] # use an averaging window to slide along the extracted high-gamma feature
winL=opt['winL'] # size of the averaging window
frameshift=opt['frameshift'] # stride of the averaging window
sf_EEG=opt['sf_EEG']
step_size=opt['step_size']
model_order=opt['model_order']
win = math.ceil(opt['win']/frameshift) # opt['win'] and frameshift are in second; win: in steps
history=math.ceil(opt['history']/frameshift) #int(opt['history']*sf_EEG) # history: in steps
stride=opt['stride']
baseline_method=opt['baseline_method']
lr=opt_transformer['lr']
norm_EEG=opt['norm_EEG']#True
norm_mel=opt['norm_mel'] #False
##################

if testing:
    stride=200
print('sid: '+str(sid)+ '; Testing mode:'+str(testing)+'; baseline_method: '+str(opt['baseline_method'])+
      '; win: '+str(win)+'; history:'+str(history)+'; stride:'+str(stride)+'; winL:'+str(winL)+'; frameshift: '+str(frameshift)+'.')

############### create result folder
the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
#result_dir=data_dir+'seq2seq_transformer/sid_'+str(sid)+'/'+ the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'/'
#result_dir=data_dir+'seq2seq_transformer/'+dataname+'/'+'mel_'+str(mel_bins)+'/sid_'+str(sid)+'/'+the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')
result_dir=data_dir+'seq2seq_transformer/'+dataname+'/'+'mel_'+str(mel_bins)+'/sid_'+str(sid)+'/'+time_stamp+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print('Result dir: '+ result_dir+'.')
writer = SummaryWriter(result_dir)
#############

continous_data=True
if dataname=='mydata':
    from speech_Ruijin.dataset import dataset_Dutch
    #x1, y1 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=1,test_shift=test_shift, mel_bins=mel_bins, continous=continous_data)
    #x2, y2 = dataset_Dutch(dataset_name=dataname, baseline_method=opt['baseline_method'], sid=sid, session=2,test_shift=300, mel_bins=mel_bins, continous=continous_data)
    x1, y1 = dataset(dataset_name=dataname, sid=sid, session=1, test_shift=test_shift, melbins=mel_bins, stacking=False, winL=winL,frameshift=frameshift)
    x2, y2 = dataset(dataset_name=dataname, sid=sid, session=2, test_shift=test_shift, melbins=mel_bins, stacking=False, winL=winL,frameshift=frameshift)
    if False:
        mu = np.mean(x1, axis=0)
        std = np.std(x1, axis=0)
        x1 = (x1 - mu) / std
        mu = np.mean(x2, axis=0)
        std = np.std(x2, axis=0)
        x2 = (x2 - mu) / std


        x=np.concatenate((x1,x2),axis=0)
        y=np.concatenate((y1,y2),axis=0)
    else:
        x=x1
        y=y1
elif dataname=='SingleWordProductionDutch':
    #x, y = get_data(dataname=dataname, sid=sid,continous_data=continous_data,mel_bins=mel_bins)  # x: (25863,127), y:(25863, 80)
    x,y=dataset(dataset_name=dataname, sid=sid, melbins=mel_bins, stacking=False, modelOrder=model_order,stepSize=step_size,winL=winL, target_SR =target_SR,
                frameshift=frameshift,use_the_official_tactron_with_waveglow=use_the_official_tactron_with_waveglow,window_eeg=window_eeg)
print('Finish reading data.')
xy_ratio = x.shape[0]/y.shape[0]
print('x,y ratio: '+str(xy_ratio)+'.')

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

if norm_EEG:
    mu=np.mean(train_x,axis=0)
    std=np.std(train_x,axis=0)
    train_x=(train_x-mu)/std

    mu = np.mean(val_x, axis=0)
    std = np.std(val_x, axis=0)
    val_x=(val_x-mu)/std

    mu = np.mean(test_x, axis=0)
    std = np.std(test_x, axis=0)
    test_x=(test_x-mu)/std
use_pca=opt['use_pca']#False
if use_pca:
    from sklearn.decomposition import PCA
    pca = PCA() #  (n_samples, n_features)
    numComps=100
    #Fit PCA to training data
    pca.fit(train_x)
    #Get percentage of explained variance by selected components
    explainedVariance =  np.sum(pca.explained_variance_ratio_[:numComps])
    #Tranform data into component space
    train_x=np.dot(train_x, pca.components_[:numComps,:].T)
    val_x = np.dot(val_x, pca.components_[:numComps,:].T)
    test_x = np.dot(test_x, pca.components_[:numComps,:].T)

# history before and after the current window
win_x, win_y, shift_x, shift_y =  (win+history)*xy_ratio, win,stride*xy_ratio,stride
x_train,y_train=fold_2d23d(train_x.transpose(),train_y[history:,:].transpose(), win_x, win_y, shift_x,shift_y)
x_val,y_val=fold_2d23d(val_x.transpose(),val_y[history:,:].transpose(), win_x, win_y, shift_x,shift_y)
x_train,y_train,x_val,y_val=[x.transpose(0,2,1) for x in (x_train,y_train,x_val,y_val)]


stride_test=stride # could be different from train/val stride
win_x, win_y, shift_x, shift_y = win+history, win* xy_ratio, stride_test, stride_test * xy_ratio
x_test,y_test=fold_2d23d(test_x.transpose(),test_y[history:,:].transpose(), win_x, win_y, shift_x,shift_y)
x_test,y_test=x_test.transpose(0,2,1),y_test.transpose(0,2,1)

# Get input_d (channel number), output_d (mel bins), timesteps from the initial dataset
input_d, output_d = x_train.shape[2], y_train.shape[2]
input_len=x_train.shape[1]
out_len = y_val.shape[1]
print('input_d: '+str(input_d)+', input length: '+str(input_len)+', output_d: '+str(output_d)+' ,output length: '+str(out_len)+'.')

from common_dl import myDataset

dataset_train = myDataset(x_train, y_train)
dataset_val = myDataset(x_val, y_val)
dataset_test = myDataset(x_test, y_test)

batch_size = opt_transformer['batch_size']
dataloader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
dataloader_val = DataLoader(dataset=dataset_val,batch_size=batch_size,shuffle=True)
batch_number=len(dataloader_train)
#batch_size=1
dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False)

opt_transformer['src_d'] = input_d # input dimension
opt_transformer['tgt_d'] = output_d # output dimension
opt_transformer['out_len'] = out_len

from pre_all import device
criterion = nn.MSELoss() # mean squared error
# setup model using hyperparameters defined above
encoder_only=False
spatial_attention=False
model = make_model(opt_transformer['src_d'],opt_transformer['tgt_d'],N=opt_transformer['Transformer-layers'],
                   d_model=opt_transformer['Model-dimensions'],d_ff=opt_transformer['feedford-size'],h=opt_transformer['headers'],
                   dropout=opt_transformer['dropout'],norm_mel=opt['norm_mel'],encoder_only=encoder_only,spatial_attention=spatial_attention).to(device)

# setup optimization function
model_opt = NoamOpt(model_size=opt_transformer['Model-dimensions'], factor=1, warmup=400,
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
#optimizer = torch.optim.Adam(model.parameters(), lr=0.015, betas=(0.9, 0.98), eps=1e-9)

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
#fig_att,ax_att=plt.subplots(figsize=(10,20))
total_epoch = 50
patients=10
train_losses = []
val_losses = []
val_teacher_force=False
# log some meta data
opt_str='win: '+str(win)+ ', stride: '+str(stride)+ ', norm_EEG: '+str(norm_EEG)+', norm_mel: '+str(norm_mel) \
        +', val_teacher_force:'+str(val_teacher_force)+', spatial_attention:'+str(spatial_attention)\
        +' encoder_only:'+str(encoder_only)+' .Batch_size:'+str(batch_size)+' .layers: '+str(opt_transformer['Transformer-layers'])+\
        ' ,model_d: '+str(opt_transformer['Model-dimensions'])+\
        ', ffd size: '+str(opt_transformer['feedford-size'])+' ,headers: '+str(opt_transformer['headers'])+' ,dropout: '+str(opt_transformer['dropout'])
experiment_description = dataname + str(sid) + ": " + opt_str
writer.add_text('Experiment description', experiment_description, 0)

from tqdm import tqdm
att_encoders, att_decoders, att_enc_decs=[],[],[] # [total_epoch,N, 64, 8, 99, 99]
for epoch in tqdm(range(total_epoch)):
    epoch = epoch + resume_epoch
    model.train()
    train_loss,pred,truth,att_encoder, att_decoder, att_enc_dec = run_epoch(epoch,batch_number,writer,data_gen(dataloader_train,encoder_only=encoder_only), model,SimpleLossCompute(model.generator, criterion, model_opt))
    # N*[8, 99, 99]-cat->[N, 8, 99, 99]
    att_encoders.append(np.concatenate(att_encoder)) # N*[8, 99, 99]
    att_decoders.append(np.concatenate(att_decoder))
    att_enc_decs.append(np.concatenate(att_enc_dec))
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

    # attention plot: att_encoders: [N, 8, 99, 99]
    ax[0].imshow(att_encoder[-1][0,:,:], cmap='RdBu_r') # -1:last encoder layer, 0:the first head
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('attention/encoder', image[0], epoch)
    ax[0].clear()

    ax[0].imshow(att_decoder[-1][0, :, :], cmap='RdBu_r')  # -1:last encoder layer, 0:the first head
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('attention/decoder', image[0], epoch)
    ax[0].clear()

    ax[0].imshow(att_enc_dec[-1][0, :, :], cmap='RdBu_r')  # -1:last encoder layer, 0:the first head
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('attention/enc_dec', image[0], epoch)
    ax[0].clear()

    print('Validating...')
    model.eval() # test the model

    if val_teacher_force: # default false
        #This is a bad validating method, because it uses teacher force like in the training step.
        with torch.no_grad():
            val_loss,pred,truth, att_encoder, att_decoder, att_enc_dec = run_epoch(epoch,batch_number,writer,data_gen(dataloader_val,encoder_only=encoder_only), model,SimpleLossCompute(model.generator, criterion, None))
            #val_loss, pred, truth = run_epoch(data_gen(dataloader_val), model,criterion, None)
            val_losses.append(val_loss)
            val_loss_avg=val_loss
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

    else:
        predictions = []
        truths = []
        val_losses=[]
        for i, (val_x, val_y) in enumerate(dataloader_val):
            # batch inference
            pred, tgt = output_prediction2(model, val_x, val_y, max_len=opt_transformer['out_len'], start_symbol=1,
                                           output_d=opt_transformer['tgt_d'],
                                           device=device, encoder_only=encoder_only)
            # the first sample of pred are all 1s
            predictions.append(pred[:,1:,:])
            truths.append(val_y)
            val_loss = criterion(pred[:,1:,:], val_y).item()
            val_losses.append(val_loss)

            writer.add_scalar('val_loss', val_loss, epoch * len(dataloader_val) + i)

        val_loss_avg = sum(val_losses)/len(val_losses)


    if epoch==resume_epoch:
        best_val_loss=val_loss_avg
        patient=patients
        best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }
        # save the best mode
        #filename = result_dir + 'best_model_epoch' + str(epoch) + '.pth'
        #torch.save(best_model, filename)
    else:
        if val_loss_avg>best_val_loss:
            patient=patient-1
        else:
            best_val_loss = val_loss_avg
            patient=patients
            best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }

    print('Epoch[{}/{}], train_loss: {:.6f},val_loss: {:.6f}. Patient: {:d}'.format(epoch, total_epoch+resume_epoch, train_loss, val_loss_avg,patient))

    # save mode every 10 epochs
    if epoch % 10 ==0:
        current_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_opt.optimizer.state_dict(),
            'loss': train_loss,
        }

        filename = result_dir + str(epoch) + '.pth'
        #torch.save(current_model, filename)


    ## plot testing result every epoch (because training is too slow)
    # not suitable for encoder_only mode; just use the validation dataset to monitor the model performance \
    # (no teacher force in encoder_only mode)
    skip_testing=True
    if not skip_testing:
        print('Testing...')
        for i, (test_x, test_y) in enumerate(dataloader_test):
            #if i > 5:
            test_x, test_y = test_x.numpy(), test_y.numpy()
            test_x, test_y = test_x[0], test_y[0]
            # make a prediction then compare it with its true output
            pred, truth = output_prediction(model, test_x, test_y, max_len=opt_transformer['out_len'], start_symbol=1,
                                            output_d=opt_transformer['tgt_d'], device=device,encoder_only=encoder_only)

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
filename=result_dir+'best_model_epoch'+str(best_model['epoch'])+'.pth'
torch.save(best_model,filename)

final_model={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_opt.optimizer.state_dict(),
            'loss': train_loss,
            }

# save and load checkpoint
filename=result_dir+str(epoch)+'_final.pth'
torch.save(final_model,filename)
#device=torch.device('cpu')
checkpoint = torch.load(filename,map_location=device)
model=model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# save attention matrix
att={}
att['att_encoders']=np.asarray(att_encoders)
att['att_decoders']=np.asarray(att_decoders)
att['att_enc_decs']=np.asarray(att_enc_decs)
filename=result_dir+'attention_matrix.npy'
np.save(filename, att)

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

