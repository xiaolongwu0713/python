'''
This script is to discriminate between speak/listen/imagine;
'''

import mne
import numpy as np
from librosa import load
import matplotlib.pyplot as plt
from tqdm import tqdm

from dSPEECH.config import *
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.deepmodel import deepnet
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
sid=1 # 1/2
sf=1024
## load epochs and sentences
filename=data_dir+'processed/'+modality+str(sid)+'/'+modality+str(sid)+'-epo.fif'
epochs=mne.read_epochs(filename)
filename2=data_dir+'processed/'+modality+str(sid)+'/sentences.npy'
sentences=np.load(filename2,allow_pickle=True)

# two events: 1 stands for 'TRIG[001]:1' and 2 stands for 'TRIG[001]:1:inserted'
epoch1=epochs['1'].get_data() # (99, 150, 15361)
epoch2=epochs['2'].get_data() # (4, 150, 15361)
epochs=np.concatenate((epoch1,epoch2),axis=0) # (103 trials, 150 channels, 15361 time)
ch_num=epochs.shape[1]

channels_to_use='unselected' # all/selected_all/selected_10/unselected

if channels_to_use=='all':
    print('Use all channels.')
    result_dir = data_dir + 'result/VAI/' + modality + str(sid) + '/'
else:
    lam=0.05
    selection_dir = data_dir + 'result/VAI_channel_selection/'+modality+str(sid)+'/'
    file=selection_dir+'selected_channels_stg_lam_'+str(lam)+'.txt'
    with open(file, 'r') as infile:
        lines=infile.read().split('\n')
    selected_channel=[int(i) for i in lines[1].split(",")]
    unselected=[i for i in range(ch_num) if i not in selected_channel]
    infoo=str(len(selected_channel))+' selected'+str(len(unselected))+' unselected.'
    selected_channel_number=len(selected_channel)

    if channels_to_use == 'selected':
        print('Use'+str(selected_channel_number)+' selected channels:' + infoo)
        result_dir = data_dir + 'result/VAI/'+modality + str(sid)+'/' + str(selected_channel_number) + 'selected_channels/'
        epochs = epochs[:, selected_channel, :]
    elif channels_to_use == 'selected_10':
        print('Use top 10 selected channels:' + infoo)
        epochs = epochs[:, selected_channel[-10:], :]
    elif channels_to_use == 'unselected':
        epochs = epochs[:, unselected, :]
        print('Use the unselected channels:' + infoo)
        result_dir = data_dir + 'result/VAI/'+ modality + str(sid)+'/'+'remove_' + str(selected_channel_number) + 'selected_channels/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# inspect the audio shows that: 500ms silence at the beginning and end of the sentence
# afile=mydriver+'/matlab/paradigms/speech_Southmead/audio/original/15_second_wavs/1.wav'
# audio, sf = load(afile,sr=None,mono=False)
# fig,ax=plt.subplots()
# plt.show()
# ax.plot(audio)
transition=0.5
listen=epochs[:,:,int(transition*sf):int((5-transition)*sf)] # (100, 150, 4096)
speak=epochs[:,:,int((5+transition)*sf):int((10-transition)*sf)] # (100, 150, 4096)
image=epochs[:,:,int((10+transition)*sf):int((15-transition)*sf)] # (100, 150, 4096)

def train_test_split(data):
    trial_number=data.shape[0]
    trial_list = list(range(trial_number))
    train_n=int(0.6*trial_number)
    val_n = int(0.2 * trial_number)
    test_n = int(0.2 * trial_number)

    test_trails=random.sample(trial_list, test_n)
    trial_number_left=np.setdiff1d(trial_list,test_trails)

    val_trails = random.sample(trial_number_left.tolist(), val_n)
    train_trails = np.setdiff1d(trial_number_left, val_trails)
    return data[train_trails,:,:], data[val_trails,:,:], data[test_trails,:,:]

train_listen,val_listen,test_listen=train_test_split(listen)
train_speak,val_speak,test_speak=train_test_split(speak)
train_image,val_image,test_image=train_test_split(image)

class_number=3
test_lists=[test_listen,test_speak,test_image]
val_lists=[val_listen,val_speak,val_image]
train_lists=[train_listen,train_speak,train_image]
wind=200
stride=70
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_lists,val_lists,test_lists,wind,stride)

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
print('Channel number: '+str(n_chans)+'.')
input_window_samples=one_window.shape[2]

lr = 0.005
weight_decay = 1e-10
batch_size = 180
epoch_num = 500
patients=5

model_name='deepnet' #'resnet'
net = deepnet(n_chans, class_number, wind)
#net=d2lresnet(class_num=class_number,end_with_logsoftmax=False) # 92%
net=net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)

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
        trainy = trainy.to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        # if cuda:
        #     loss = criterion(y_pred, trainy.squeeze().cuda().long())
        # else:
        #     loss = criterion(y_pred, trainy.squeeze())
        loss = criterion(y_pred, trainy.squeeze().long())
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.squeeze() == trainy.squeeze())
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
                val_y = val_y.to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += torch.sum(preds.squeeze() == val_y.squeeze())

        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        print("Training loss:{:.2f},Accuracy:{:.2f}; Evaluation accuracy:{:.2f}.".format(train_loss, train_acc,val_acc))
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
        test_y = test_y.to(device)
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)

        running_corrects += torch.sum(preds.squeeze() == test_y.squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '_test_acc.txt'
with open(filename,'w') as f:
    f.write("Test accuracy: {:.2f}.".format(test_acc))
    f.write('\n')

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename,allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])




