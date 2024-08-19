import numpy as np
from HaiMeiKang.config import *
import pandas
import mne
from torch.optim import lr_scheduler
from HaiMeiKang.d2l_resnet import d2lresnet

result_path=data_dir + 'from_hospital/'
print('Reading data.')
raws=[]
for i in range(10):
    raws.append(mne.io.read_raw_fif(data_dir+'from_hospital/'+str(i)+'.fif'))

list_of_epochs = []
for raw in raws:
    event = mne.find_events(raw, stim_channel="stim")
    epoch = mne.Epochs(raw, event, tmin=0, tmax=5, baseline=None)
    list_of_epochs.append(epoch.get_data())

trial_numbers = [list(range(epochi.shape[0])) for epochi in list_of_epochs]

test_trials = [random.sample(trial_number, 4) for trial_number in trial_numbers]
trial_numbers_left = [np.setdiff1d(trial_numbers[i], test_trials[i]).tolist() for i in range(len(trial_numbers))]
val_trials = [random.sample(trial_number, 4) for trial_number in trial_numbers_left]
train_trials = [np.setdiff1d(trial_numbers_left[i], val_trials[i]).tolist() for i in range(len(trial_numbers))]

test_epochs = [epochi[test_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]
val_epochs = [epochi[val_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]
train_epochs = [epochi[train_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]

from gesture.utils import windowed_data

print('Windowing data.')
gen_data_all = None
X_train, y_train, X_val, y_val, X_test, y_test = windowed_data(train_epochs, val_epochs, test_epochs, 1000, 500,
                                                               gen_data_all=gen_data_all, train_mode='original',
                                                               method=None)


train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 32
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

model_name='resnet'
net=d2lresnet(class_num=10,end_with_logsoftmax=False) # 92%
net=net.to(device)

lr = 0.01
weight_decay = 1e-10
batch_size = 32
epoch_num = 500
patients=10


criterion = torch.nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

train_losses=[]
train_accs=[]
val_accs=[]
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for epoch in range(epoch_num):
    print("------ epoch" + str(epoch) + "-----")
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(train_loader):
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
        else:
            trainx = trainx.float()
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred, trainy.squeeze().cuda().long())
        else:
            loss = criterion(y_pred, trainy.squeeze())


        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.squeeze())
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
            for _, (val_x, val_y) in enumerate(val_loader):
                if (cuda):
                    val_x = val_x.float().cuda()
                    # val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    # val_y = val_y.float()
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())

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
        savepath = result_path + 'checkpoint_'+model_name+'_' + str(best_epoch) + '.pth'
        torch.save(state, savepath)
        break

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(test_loader):
        if (cuda):
            test_x = test_x.float().cuda()
            # val_y = val_y.float().cuda()
        else:
            test_x = test_x.float()
            # val_y = val_y.float()
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)

        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

filename=result_path+'result.txt'
with open(filename,'w') as f:
    f.write("Test accuracy: {:.2f}.".format(test_acc))
    f.write('\n')

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

filename=result_path+'result.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename+'.npy',allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])


