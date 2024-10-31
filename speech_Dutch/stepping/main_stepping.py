import io
import sys
import socket
import PIL
from torchvision.transforms import ToTensor
from tqdm import tqdm

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

from gesture.models.d2l_resnet import d2lresnet
from speech_Dutch.stepping.dataset import dataset_stepping
from speech_Dutch.stepping.model_stepping import tsception
from datetime import datetime
import pytz
from speech_Dutch.config import *
from common_dl import *
from torch.utils.data import DataLoader
from speech_Dutch.dataset import dataset_Dutch
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
set_random_seeds(999)

resume=False
if computer=='mac':
    testing=True
elif computer=='workstation' and debugging:
    testing=True
elif computer=='workstation' and not debugging:
    testing=False

dataname='SingleWordProductionDutch'
if running_from_CMD: # run from cmd on workstation
    sid = int(float(sys.argv[1]))
else:
    sid = 3
if dataname=='SingleWordProductionDutch':
    pt_folder='sub-' + f"{sid:02d}"
    from speech_Dutch.stepping.opt import opt_SingleWordProductionDutch as opt
else:
    pt_folder='sid'+str(sid)
    from speech_Dutch.stepping.opt import opt_mydata as opt
mel_bins=opt['mel_bins']
sf_EEG=opt['sf_EEG']

the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
print("Testing: "+str(testing)+".")
if testing or debugging:
    batch_size = 3
    wind_size = 200
    stride = 400
    epoch_num = 1
    patients = 2
    result_dir = data_dir + 'stepping/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(sid) + '_wind_' + str(
        wind_size) + '/' + the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'_testing/'
else:
    batch_size=opt['batch_size']
    wind_size = opt['wind_size']
    stride = opt['stride']
    epoch_num = opt['epoch_num']
    patients = opt['patients']
    result_dir = data_dir + 'stepping/' + dataname + '/' + 'mel_' + str(mel_bins) + '/sid_' + str(sid) + '_wind_'+\
                 str(wind_size)+'/' + the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'/'
if not os.path.exists(result_dir):
    print('Result_dir: '+result_dir+'.')
    os.makedirs(result_dir)
writer = SummaryWriter(result_dir)

shift=None #0.03 # -0.03:left shift 30 ms; -30/-60/-90/
train_ds, val_ds, test_ds = dataset_stepping(dataname=dataname,sid=sid,wind_size=wind_size,
                                             stride=stride, mel_bins=mel_bins,shift=shift,test_only=False)

train_size=len(train_ds)
val_size=len(val_ds)
test_size=len(test_ds)
print('Train size: '+str(train_size)+', val size: '+str(val_size)+', test size: '+str(test_size)+'.')
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

ch_num=train_ds[0][0].shape[0]
learning_rate=opt['learning_rate'] #0.005
dropout=opt['dropout']
net_name=opt['net_name']
if net_name=='tsception':
    net=tsception(1000,ch_num,wind_size,6,6,dropout,mel_bins)
elif net_name=='resnet':
    net=d2lresnet(task='regression',reg_d=mel_bins)
net=net.to(device)
#optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate,weight_decay=1e-4)
loss_fun = nn.MSELoss()

if resume:
    resume_epoch=50
    filename = 'H:/Long/data/speech_Ruijin/result/P1/win_400/2023_06_17_12_48/checkpoint_49.pth'
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    resume_epoch=0

fig,ax=plt.subplots(2,1, figsize=(6,4))
train_losses=[]
val_losses=[]
for epoch in tqdm(range(epoch_num)):
    epoch=epoch+resume_epoch
    print('Training on sid: '+str(sid)+'; Epoch:'+str(epoch)+'.')
    net.train()
    y_preds=[]
    running_loss = 0.0
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x=x.float().to(device)
        y=y.float().to(device)
        y_pred= net(x) # x:torch.Size([3 batch, 118channel, 200time])
        loss = loss_fun(y_pred,y) # y: torch.Size([3, 80])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.shape[0]
        if batch==len(train_loader)-2:
            plot_y=y
            plot_y_pred=y_pred
        if testing:
            break
    train_loss = running_loss / train_size
    train_losses.append(train_loss)
    #show_batch=min(60,y.shape[0])
    ax[0].imshow(plot_y.cpu().numpy().squeeze().transpose(), cmap='RdBu_r',aspect='auto')
    ax[1].imshow(plot_y_pred.cpu().detach().numpy().squeeze().transpose(), cmap='RdBu_r',aspect='auto')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('Image/train', image[0], epoch)
    ax[0].clear()
    ax[1].clear()

    print('Evaluating....')
    running_loss = 0.0
    truths = []
    preds = []
    if epoch % 1 == 0:
        net.eval()
        with torch.no_grad():
            for batch, (val_x, val_y) in enumerate(val_loader):
                val_x=val_x.float().to(device)
                val_y=val_y.float().to(device)
                y_pred = net(val_x)  # x:torch.Size([3 batch, 118channel, 200time])
                loss = loss_fun(y_pred, val_y)  # y: torch.Size([3, 80])
                truths.append(val_y.cpu().numpy().squeeze())
                preds.append(y_pred.cpu().numpy().squeeze())

                running_loss += loss.item() * val_x.shape[0]
                if batch==len(val_loader)-2:
                    plot_valy=val_y
                    plot_valy_pred=y_pred
            if testing:
                break
            val_loss = running_loss / val_size
            val_losses.append(val_loss)
        print("Training loss:{:.2f}; Evaluation loss:{:.2f}.".format(train_loss, val_loss))
        truths = np.concatenate(truths, axis=0)
        preds = np.concatenate(preds, axis=0)

    ax[0].imshow(truths.transpose(), cmap='RdBu_r',aspect='auto')
    ax[1].imshow(preds.transpose(), cmap='RdBu_r',aspect='auto')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('Image/val', image[0], epoch)
    ax[0].clear()
    ax[1].clear()

    if epoch==resume_epoch:
        best_epoch=resume_epoch
        best_loss=val_loss
        patient=patients
        best_model = {
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
            best_model = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'loss': epoch_loss
            }
            #filename = result_dir + 'best_model_epoch' + str(epoch) + '.pth'
            #torch.save(best_model, filename)
        else:
            patient=patient-1
    print("patients left: {:d}".format(patient))
    if patient==0:
        break
    if testing:
        #pass # test through all epochs
        break
filename=result_dir+'best_model_epoch'+str(best_model['epoch'])+'.pth'
torch.save(best_model,filename)


