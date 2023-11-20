import sys
import socket
import time

from tqdm import tqdm

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
from datetime import datetime
import pytz
from speech_Ruijin.config import *
from speech_Ruijin.model import Seq2SeqEncoder2, Seq2SeqAttentionDecoder, EncoderDecoder
from common_dl import *
from speech_Ruijin.seq2seq.dataset_seq2seq import dataset_seq2seq
from torch.utils.data import DataLoader

testing=False

sf_EEG=1000
sf_audio=48000
mel_ration=255
sid=1

the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
if testing:
    result_dir = data_dir+'result/' + the_time.strftime('%Y_%m_%d')+'_'+the_time.strftime('%H_%M') + '_testing/'
else:
    result_dir = data_dir+'result/' + the_time.strftime('%Y_%m_%d')+'_'+the_time.strftime('%H_%M') + '/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

train_ds,val_ds,test_ds=dataset_seq2seq(sid)
train_size=len(train_ds)
val_size=len(val_ds)
test_size=len(test_ds)
batch_size=1 # sequence in a batch is not the same
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

ch_num=train_ds[0][0].shape[0]
in_features=ch_num
out_features=80
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
#source_len=1000
#tgt_len=500
encoder = Seq2SeqEncoder2(in_features, embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(out_features, embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder).to(device)

# shapes of src,target and out: (time_steps, batch_size, features)
loss_fun=torch.nn.MSELoss()
learning_rate=0.0001
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

epoch_num=20
patients=5
train_losses=[]
val_losses=[]
test_losses=[]
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for epoch in range(epoch_num):
    print('Training on sid: '+str(sid)+'; Epoch:'+str(epoch)+'.')
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        src = trainx.float().transpose(1, 2).transpose(0, 1).to(device)
        tgt = trainy.float().transpose(0,1).to(device) # torch.Size([295, 1, 80])

        optimizer.zero_grad()
        y_pred = net(src,tgt) # torch.Size([147, 1, 80])
        loss = loss_fun(y_pred, tgt)
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]

        if testing:
            break
    #print("train_size: " + str(train_size))
    #lr_scheduler.step() # test it
    train_loss = running_loss / train_size
    train_losses.append(train_loss)

    running_loss = 0.0
    if epoch % 1 == 0:
        net.eval()
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
                src = val_x.float().transpose(1, 2).transpose(0, 1).to(device)
                tgt = val_y.float().transpose(0,1).to(device)
                #out_len=int(src.shape[0]*sf_audio/sf_EEG/mel_ration)
                out_len=tgt.shape[0]
                outputs,attention_weights = net.predict_step(src, out_len) # torch.Size([1, 194, 80])
                loss = loss_fun(outputs, tgt)
                running_loss += loss.item() * val_x.shape[0]

                if testing:
                    break
            val_loss = running_loss / val_size
            val_losses.append(val_loss)
        print("Training loss:{:.2f}; Evaluation loss:{:.2f}.".format(train_loss, val_loss))
    if epoch==0:
        best_epoch=0
        best_loss=val_loss
        patient=patients
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            # 'loss': epoch_loss
        }
    else:
        if val_loss<best_loss:
            best_epoch=epoch
            best_loss=val_loss
            patient=patients
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'loss': epoch_loss
            }

        else:
            patient=patient-1
    print("patients left: {:d}".format(patient))
    if patient==0:
        savepath = result_dir + 'checkpoint_'+str(best_epoch) + '.pth'
        torch.save(state, savepath)
        break
    if testing:
        pass # test through all epochs
        #break

if not testing:
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
test_preds=[]
test_tgts=[]
# print("Validating...")
with torch.no_grad():
    running_loss = 0.0
    for _, (test_x, test_y) in enumerate(test_loader):
        src = test_x.float().transpose(1, 2).transpose(0, 1).to(device)
        tgt = test_y.float().transpose(0,1).to(device)
        test_tgts.append(tgt.squeeze().numpy())
        #out_len=int(src.shape[0]*sf_audio/sf_EEG/mel_ration)
        out_len = tgt.shape[0]
        pred,attention_weights = net.predict_step(src,out_len) #pred: torch.Size([1, 135, 80])
        test_preds.append(pred.squeeze().numpy())
        #outputs = net(test_x)
        loss = loss_fun(pred, tgt)
        running_loss += loss.item() * test_x.shape[0]
    test_loss = running_loss / test_size
    print("Test loss: {:.2f}.".format(test_loss))

fig, ax = plt.subplots(2,1)
for i in range(len(test_preds)):
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(test_preds[i].transpose(),origin='lower',cmap='RdBu_r')
    ax[0].set_title('Predicted')
    ax[1].imshow(test_tgts[i].transpose(), origin='lower', cmap='RdBu_r')
    ax[1].set_title('Truth')
    #fig.tight_layout()
    figname = result_dir + 'test_pred' + str(epoch) + str(i)+ '.png'
    fig.savefig(figname)
    time.sleep(0.005)
    #plt.close(fig)
if not testing:
    train_result = {}
    train_result['train_losses'] = train_losses
    train_result['val_losses'] = val_losses
    train_result['test_losses'] = test_loss
    filename=result_dir + 'checkpoint_'+str(best_epoch) + '.npy'
    np.save(filename,train_result)

'''
## check the length scale ration between waveform and mel (255)
trainx,trainy=next(iter(train_loader))
trainy=trainy.float()
trainy /= torch.max(torch.abs(trainy))
mel = mel_transformer.mel_spectrogram(trainy).squeeze()
src = trainx.type(FloatTensor).transpose(1, 2).transpose(0, 1).to(device)
#src = trainx.to(dtype=torch.float32).transpose(1, 2).transpose(0, 1).to(device)
tgt = mel.float().transpose(0, 1)[:, None, :].to(device)

src.shape[0]*sf_audio/sf_EEG/mel_ration
'''

def aaa(a,b,*args):
    print(a)
    print(b)
    print(args)

inn=[1,2,3]
aaa(*inn,4)
