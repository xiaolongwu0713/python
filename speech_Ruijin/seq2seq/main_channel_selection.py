'''
sid 3: 82
Took too long time. use the baseline linear regression instead.
'''
import sys
import socket

from scipy.stats import pearsonr

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import io
import time
import PIL
from torchvision.transforms import ToTensor
from tqdm import tqdm
import math

from example_speech.word_decoding_HD_ECoG_main.src.models.seq2seq import make_model
from speech_Ruijin.utils import fold_2d23d
from datetime import datetime
import pytz
from speech_Ruijin.config import *
from speech_Ruijin.seq2seq.model_d2l import Seq2SeqEncoder2, Seq2SeqAttentionDecoder, EncoderDecoder
from common_dl import *
from speech_Ruijin.seq2seq.dataset_seq2seq import dataset_seq2seq
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from speech_Ruijin.seq2seq.opt import opt_SingleWordProductionDutch as opt
from speech_Ruijin.seq2seq.opt import channel_numbers
from pre_all import set_random_seeds
set_random_seeds(99999)

dataname='SingleWordProductionDutch'
if running_from_CMD: # run from cmd on workstation
    sid = int(float(sys.argv[1]))
else:
    sid = 3
channel_number=channel_numbers[sid]
###############
mel_bins=opt['mel_bins']
winL=opt['winL']
frameshift=opt['frameshift']
win = math.ceil(opt['win']/frameshift) # int: steps
history=math.ceil(opt['history']/frameshift) #int(opt['history']*sf_EEG) # int: steps
stride=opt['stride']
stride_test = opt['stride_test']
baseline_method=opt['baseline_method']
norm_EEG=opt['norm_EEG']#True
norm_mel=opt['norm_mel'] #False
sf_EEG=opt['sf_EEG']

embed_size=opt['embed_size'] #= 256
num_hiddens=opt['num_hiddens'] #= 256
num_layers=opt['num_layers'] #= 2
dropout=opt['dropout'] #= 0.5
lr=opt['lr']
batch_size = opt['batch_size']
##################

resume=False
testing=False
testing=(testing or debugging or computer=='mac')
if testing or debugging:
    batch_size = 3
else:
    batch_size = 128
print('sid: '+str(sid)+ '; Testing:'+str(testing)+'. wind:'+str(win)+', stride:'+str(stride)+', history:'+str(history)+'.')

############### create result folder
the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
result_dir=data_dir+'seq2seq_channel_selection/'+dataname+'/'+'mel_'+str(mel_bins)+'/sid_'+str(sid)+'/'\
           +the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')
if testing:
    result_dir=result_dir+'_testing/'
else:
    result_dir=result_dir+'/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print('Result dir: '+ result_dir+'.')
writer = SummaryWriter(result_dir)
######################
all_channels=list(range(channel_number))
opt_channels=[]
channel_tobe_selected=5
for i in range(channel_tobe_selected):
    text_file = result_dir + 'result.txt'
    with open(text_file, 'a') as f:
        f.write("########### select the "+str(i)+"th channel ###########")
        f.write('\n')
    channels_left=[i for i in all_channels if i not in opt_channels]
    mses = []
    for j,channel in enumerate(channels_left):
        use_channels=opt_channels.copy()
        use_channels.append(channel)
        print('Testing channels: ' +','.join(str(i) for i in use_channels))

        ######## get mse value #######
        from speech_Ruijin.baseline_linear_regression.extract_features import dataset
        x, y = dataset(dataset_name=dataname, sid=sid, use_channels=use_channels, melbins=mel_bins, stacking=False,winL=winL, frameshift=frameshift)
        xy_ratio = y.shape[0] / x.shape[0]
        print('x,y ratio: ' + str(xy_ratio) + '.')

        lenx = x.shape[0]
        leny = y.shape[0]
        train_x = x[:int(lenx * 0.8), :]  # (246017, 127)
        val_x = x[int(lenx * 0.8):int(lenx * 0.9), :]
        test_x = x[int(lenx * 0.9):, :]
        train_y = y[:int(leny * 0.8), :]
        val_y = y[int(leny * 0.8):int(leny * 0.9), :]
        test_y = y[int(leny * 0.9):, :]

        if norm_EEG:
            mu = np.mean(train_x, axis=0)
            std = np.std(train_x, axis=0)
            train_x = (train_x - mu) / std

            mu = np.mean(val_x, axis=0)
            std = np.std(val_x, axis=0)
            val_x = (val_x - mu) / std

            mu = np.mean(test_x, axis=0)
            std = np.std(test_x, axis=0)
            test_x = (test_x - mu) / std

        win_x, win_y, shift_x, shift_y = int(win + history), int(win * xy_ratio), int(stride), int(stride * xy_ratio)
        x_train, y_train = fold_2d23d(train_x.transpose(), train_y[history:, :].transpose(), win_x, win_y, shift_x,shift_y)
        x_val, y_val = fold_2d23d(val_x.transpose(), val_y[history:, :].transpose(), win_x, win_y, shift_x, shift_y)
        stride_test = stride  # could be different from train/val stride
        shift_x, shift_y = int(stride_test), int(stride_test * xy_ratio)
        x_test, y_test = fold_2d23d(test_x.transpose(), test_y[history:, :].transpose(), win_x, win_y, shift_x, shift_y)

        dataset_train = myDataset(x_train, y_train)
        dataset_val = myDataset(x_val, y_val)
        dataset_test = myDataset(x_test, y_test)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        batch_number = len(dataloader_train)

        train_size=len(dataset_train)
        val_size=len(dataset_val)
        test_size=len(dataset_test)

        ch_num=dataset_train[0][0].shape[0]
        in_features=ch_num
        out_len=dataset_train[0][1].shape[1]
        out_features=dataset_train[0][1].shape[0]
        #device = torch.device('cpu') # GPU is too slow, because of the GRU?
        #source_len=1000
        #tgt_len=500
        using_d2l=True
        if using_d2l:
            embedding=True
            bidirectional=True # not implemented
            if not embedding:
                num_hiddens=out_features
            encoder = Seq2SeqEncoder2(in_features, embed_size, num_hiddens, num_layers, dropout)
            decoder = Seq2SeqAttentionDecoder(out_features, embed_size, num_hiddens, num_layers, dropout=dropout,batch_size=batch_size)
            net = EncoderDecoder(encoder, decoder).to(device)
        else:
            n_in=ch_num
            n_out=out_features
            #out_len=500
            drop_ratio=0.5
            seq_n_enc_layers=2
            seq_n_dec_layers=2
            seq_bidirectional=False
            #device = torch.device("cpu")
            net = make_model(in_dim=n_in,out_dim=n_out,out_len=out_len,drop_ratio=drop_ratio,n_enc_layers=seq_n_enc_layers,
                             n_dec_layers=seq_n_dec_layers,enc_bidirectional=seq_bidirectional).to(device)

        # log some meta data
        opt_str='win: '+str(win)+ ', stride: '+str(stride)+ ', norm_EEG: '+str(norm_EEG)+', norm_mel: '+str(norm_mel) \
                +' .Batch_size:'+str(batch_size)+' .embed_size: '+str(opt['embed_size'])+\
                ' ,num_hiddens: '+str(opt['num_hiddens'])+', num_layers: '+str(opt['num_layers'])+\
                ' ,dropout: '+str(opt['dropout'])+' ,learn rate: '+str(opt['lr'])
        experiment_description = dataname + str(sid) + ": " + opt_str
        writer.add_text('Experiment description', experiment_description, 0)

        # shapes of src,target and out: (time_steps, batch_size, features)
        loss_fun=torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr)

        if resume:
            resume_epoch=50
            filename = 'H:/Long/data/speech_Ruijin/result/P1/win_400/2023_06_17_12_48/checkpoint_49.pth'
            checkpoint = torch.load(filename)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            resume_epoch=0

        train_losses=[]
        val_losses=[]
        test_losses=[]
        reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        fig,ax=plt.subplots(2,1)
        epoch_num=100
        patients=5
        for epoch in tqdm(range(epoch_num)):
            epoch=epoch+resume_epoch
            print('Training on sid: '+str(sid)+'; Epoch:'+str(epoch)+'.')
            net.train()
            loss_epoch = 0
            running_loss = 0.0
            y_preds=[]
            for i, (trainx, trainy) in enumerate(dataloader_train):
                src = trainx.float().permute(2,0,1).to(device) #(time,batch,feature)
                tgt = trainy.float().permute(2,0,1).to(device) # torch.Size([295, 1, 80])

                optimizer.zero_grad()
                y_pred,_ = net(src,tgt[1:,:,:]) # torch.Size([147time, 1batch, 80])
                loss = loss_fun(y_pred, tgt[1:,:,:])
                loss.backward()  # calculate the gradient and store in .grad attribute.
                optimizer.step()
                running_loss += loss.item() * trainx.shape[0]
                y_preds.append(y_pred)
                writer.add_scalar('train_loss', loss.item(), epoch * len(dataloader_train) + i)
                if testing:
                    break
            #print("train_size: " + str(train_size))
            #lr_scheduler.step() # test it
            train_loss = running_loss / train_size
            train_losses.append(train_loss)

            ax[0].imshow(tgt.cpu()[:,-1,:].numpy().squeeze(), cmap='RdBu_r', aspect='auto')
            ax[1].imshow(y_pred.cpu()[:,-1,:].detach().numpy().squeeze(),cmap='RdBu_r', aspect='auto')
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            writer.add_image('Image/train', image[0], epoch)
            ax[0].clear()
            ax[1].clear()
            #pic=vutils.make_grid(torch.concatenate(y_preds,axis=1).permute(1,0,2)[:,None,:,:], normalize=True, scale_each=True)
            #writer.add_image('Image/train', pic, epoch) # B x C x H x W

            print('Evaluating....')
            running_loss = 0.0
            outputs=[]
            if epoch % 1 == 0:
                net.eval()
                with torch.no_grad():
                    for i, (val_x, val_y) in enumerate(dataloader_val):
                        src = val_x.float().permute(2,0,1).to(device)
                        tgt = val_y.float().permute(2,0,1).to(device)
                        #src,tgt=src.permute(1,0,2),tgt.permute(1,0,2) # ECoG_seq2seq
                        out_len=tgt.shape[0]
                        # no teacher force during validation
                        output,attention_weights = net.predict_step(src, tgt[1:,:,:],out_len) # torch.Size([1, 194, 80])
                        loss = loss_fun(output, tgt[1:,:,:])
                        running_loss += loss.item() * val_x.shape[0]
                        outputs.append(output)
                        if testing:
                            pass
                        writer.add_scalar('val_loss', loss.item(), epoch * len(dataloader_val) + i)
                    val_loss = running_loss / val_size
                    val_losses.append(val_loss)

                    # writer.add_image('Image/val', pic, epoch)
                    ax[0].imshow(tgt.cpu()[:, -1, :].numpy().squeeze(), cmap='RdBu_r', aspect='auto')
                    ax[1].imshow(output.cpu()[:, -1, :].detach().numpy().squeeze(), cmap='RdBu_r', aspect='auto')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='jpeg')
                    buf.seek(0)
                    image = PIL.Image.open(buf)
                    image = ToTensor()(image).unsqueeze(0)
                    writer.add_image('Image/val', image[0], epoch)
                    ax[0].clear()
                    ax[1].clear()

                print("Training loss:{:.2f}; Evaluation loss:{:.2f}.".format(train_loss, val_loss))

            if epoch==resume_epoch:
                best_epoch=resume_epoch
                best_loss=val_loss
                patient=patients
                best_model = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'hyperp': experiment_description
                }

            else:
                if val_loss<best_loss:
                    best_epoch=epoch
                    best_loss=val_loss
                    patient=patients
                    best_model = {
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'hyperp': experiment_description
                    }

                else:
                    patient=patient-1

            current_model = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'loss': epoch_loss
            }
            filename_current = result_dir + 'epoch' + str(epoch) + '.pth'
            #torch.save(current_model, filename_current)

            print("patients left: {:d}".format(patient))
            if patient==0:
                break
            if testing:
                #pass # test through all epochs
                break
        filename_best = result_dir + 'best_model_epoch' + str(best_model['epoch']) + '.pth'
        #torch.save(best_model, filename_best)
        final_model={
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hyperp': experiment_description
                    }
        filename_final=result_dir+str(epoch)+'_final.pth'
        #torch.save(final_model,filename_final)

        ## inference
        from speech_Ruijin.transformer.utils import test_seq2seq_model, averaging

        #checkpoint = torch.load(filename_best, map_location=device)
        checkpoint=best_model
        net.load_state_dict(checkpoint['net'])
        del checkpoint
        pred, tgt = test_seq2seq_model(net, dataloader_test, result_dir)

        tgt = np.concatenate((tgt[:, 0, :][:, np.newaxis, :], tgt), axis=1)
        pred = np.concatenate((pred[:, 0, :][:, np.newaxis, :], pred), axis=1)

        filename = result_dir + '/prediction.npy'
        np.save(filename, pred)
        filename = result_dir + '/truth.npy'
        np.save(filename, tgt)

        avg_tgt = averaging(tgt, win_y, shift_y)  # (30750, 40)
        avg_pred = averaging(pred, win_y, shift_y)  # (6121, 100, 40)-->(6220, 40)


        ax[0].imshow(avg_tgt.transpose()[:,10:-10],aspect='auto')
        ax[1].imshow(avg_pred.transpose()[:,10:-10],aspect='auto')
        figname=result_dir+'plot.png'
        fig.savefig(figname)
        ax[0].clear()
        ax[1].clear()

        avg_tgt=avg_tgt[10:-10,:]
        avg_pred=avg_pred[10:-10,:]
        #avg_tgt=avg_tgt.transpose()
        #avg_pred=avg_pred.transpose()
        partial_teste=True # only test partial testing data in the original paper
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

        text_file = result_dir + 'result.txt'
        with open(text_file, 'a') as f:
            f.write('corr:'+str(mean_corr))
            f.write('\n')
            f.write('mse:'+str(loss.item()))

        mses.append(loss.item())

    best = mses.index(max(mses))
    best_channel = channels_left[best]
    with open(text_file, 'a') as f:
        f.write('selected channel:' + str(best_channel))
    opt_channels.append(best_channel)

with open(text_file, 'a') as f:
    for item in opt_channels:
        f.write("%s , " % str(item))




