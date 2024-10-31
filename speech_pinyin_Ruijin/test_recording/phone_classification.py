'''
This script is to classify phonemes: a/e/i/u
'''
import glob
import matplotlib
from braindecode.models import EEGNetv4, ShallowFBCSPNet

from gesture.models.deepmodel import deepnet

matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from speech_pinyin_Ruijin.config import *
from dSPEECH.phoneme.util import train_test_split, wind_list_of_2D
from gesture.models.d2l_resnet import d2lresnet_simple, d2lresnet
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
task='pinyin'
sid=2 #
sf=1000

folder=data_dir+str(sid)+'-*'
folder=os.path.normpath(glob.glob(folder)[0])
folder=folder.replace("\\", "/")
result_dir=folder+'/result/'
filename=folder+'/processed/'+task+'.fif'
raw=mne.io.read_raw_fif(filename,preload=True)
info=raw.info

# normalization helps
#raw.apply_function(lambda x: (x - np.mean(x) / np.std(x)))
if 1==1:
    stim=raw.get_data(picks=['stim'])
    data=raw.get_data(picks=['eeg',]).transpose()
    scaler = StandardScaler()
    data2 = scaler.fit_transform((data))
    raw = mne.io.RawArray(np.concatenate((data2.transpose(),stim),axis=0), info)

events=mne.find_events(raw, stim_channel='Trigger')
raw.drop_channels(['Trigger'])
events=events[3:-1,:]
events_tasks=np.asarray([tmp for tmp in events if tmp[-1] != 99])
epochs_all = mne.Epochs(raw, events_tasks, tmin=3.5, tmax=5,baseline=None)
epoch1=epochs_all['1'].get_data() # (16, 118, 1001)
epoch2=epochs_all['2'].get_data()
epoch3=epochs_all['3'].get_data()
epoch4=epochs_all['4'].get_data()

if 1==0:
    # normalization here
    def norm_epoch(epoch):
        for i in range(len(epoch)):
            ee=epoch[i]
            tmp=ee.transpose()
            scaler = StandardScaler()
            tmp = scaler.fit_transform((tmp))
            epoch[i]=tmp.transpose()
        return epoch
    epoch1=norm_epoch(epoch1)
    epoch2=norm_epoch(epoch2)
    epoch3=norm_epoch(epoch3)
    epoch4=norm_epoch(epoch4)

train1,val1,test1=train_test_split(epoch1) # 12/2/2,list of (118,1001);
train2,val2,test2=train_test_split(epoch2)
train3,val3,test3=train_test_split(epoch3)
train4,val4,test4=train_test_split(epoch4)

wind=200
stride=10
train1,val1,test1=wind_list_of_2D(train1,wind, stride),wind_list_of_2D(val1,wind, stride),wind_list_of_2D(test1,wind, stride)
train2,val2,test2=wind_list_of_2D(train2,wind, stride),wind_list_of_2D(val2,wind, stride),wind_list_of_2D(test2,wind, stride)
train3,val3,test3=wind_list_of_2D(train3,wind, stride),wind_list_of_2D(val3,wind, stride),wind_list_of_2D(test3,wind, stride)
train4,val4,test4=wind_list_of_2D(train4,wind, stride),wind_list_of_2D(val4,wind, stride),wind_list_of_2D(test4,wind, stride)

X_train=np.asarray(train1+train2+train3+train4) # (864, 118, 200)
X_val=np.asarray(val1+val2+val3+val4) # (432, 118, 200)
X_test=np.asarray(test1+test2+test3+test4)

y_train=[0,]*len(train1)+[1,]*len(train2)+[2,]*len(train3)+[3,]*len(train4)
y_val=[0,]*len(val1)+[1,]*len(val2)+[2,]*len(val3)+[3,]*len(val4)
y_test=[0,]*len(test1)+[1,]*len(test2)+[2,]*len(test3)+[3,]*len(test4)

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 20 # larger batch_size slows the training
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190

# Extract number of chans and time steps from dataset
one_window=next(iter(train_loader))[0]
n_chans = one_window.shape[1]
input_window_samples=one_window.shape[2]

lr = 0.005
weight_decay = 1e-10
epoch_num = 500
patients=10

model_name='resnet'
class_number=4
#net = ShallowFBCSPNet(n_chans,class_number,input_window_samples=input_window_samples,final_conv_length='auto',) #18%
#net = EEGNetv4(n_chans, class_number, input_window_samples=input_window_samples, final_conv_length='auto', drop_prob=0.5) #32
net=deepnet(n_chans,class_number,wind) # 33%
#net=d2lresnet_simple(class_num=class_number,end_with_logsoftmax=False) #34
#net=d2lresnet(class_num=class_number,end_with_logsoftmax=False) # 28%
net=net.to(device)
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr) # result not good: evaluation accuracy is always chance level (25%)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)


def correct_count(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    #acc = correct_results_sum / y_test.shape[0]
    #acc = torch.round(acc * 100)
    return correct_results_sum

train_losses=[]
train_accs=[]
val_accs=[]
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for epoch in range(epoch_num):
    print("------ epoch" + str(epoch) + ': sid' +str(sid)+' using '+model_name+'-----')
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        trainx = trainx.float().to(device)
        trainy = trainy.type(torch.LongTensor).to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        # if cuda:
        #     loss = criterion(y_pred, trainy.squeeze().cuda().long())
        # else:
        #     loss = criterion(y_pred, trainy.squeeze())
        loss = criterion(y_pred, trainy.squeeze())
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        #loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.cpu().squeeze())
        #running_corrects += correct_count(y_pred.squeeze(),trainy.squeeze()) #torch.sum(preds.squeeze() == trainy.squeeze())
    #print("train_size: " + str(train_size))
    #lr_scheduler.step() # test it
    train_loss = running_loss / train_size
    train_acc = (running_corrects.double() / train_size).item()
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    #print("Training loss: {:.2f}; Accuracy: {:.2f}.".format(train_loss,train_acc))
    #print("Training " + str(epoch) + ": loss: " + str(epoch_loss) + "," + "Accuracy: " + str(epoch_acc.item()) + ".")

    running_loss = 0.0
    running_corrects = 0
    if epoch % 1 == 0:
        net.eval()
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(tqdm(val_loader)):
                val_x = val_x.float().to(device)
                val_y = val_y.type(torch.LongTensor).to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                #running_corrects += correct_count(outputs.squeeze(),val_y.squeeze())#torch.sum(preds.squeeze() == val_y.squeeze())
                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.cpu().squeeze())

        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        print("Training loss:{:.4f},Accuracy:{:.4f}; Evaluation accuracy:{:.4f}.".format(train_loss, train_acc,val_acc))
    if epoch==0:
        best_epoch=0
        best_acc=val_acc
        patient=patients
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            # 'loss': epoch_loss
        }
    else:
        if val_acc>best_acc:
            best_epoch=epoch
            best_acc=val_acc
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
        savepath = result_dir + 'checkpoint_'+model_name+'_' + str(best_epoch) + '.pth'
        torch.save(state, savepath)
        break

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(tqdm(test_loader)):
        test_x = test_x.float().to(device)
        test_y = test_y.type(torch.LongTensor).to(device)
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)

        #running_corrects += correct_count(outputs.squeeze(),test_y.squeeze())# torch.sum(preds.squeeze() == test_y.squeeze())
        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.cpu().squeeze())

test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.4f}.".format(test_acc))

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '_test_acc.txt'
with open(filename,'w') as f:
    f.write("Test accuracy: {:.4f}.".format(test_acc))
    f.write('\n')

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

print('Saving decoding result to: '+result_dir+'.')
filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename,allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])




