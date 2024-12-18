'''
This scirpt tries to validate the code used in phoneme classification of SEEG data collected from Ruijing Hospital.
The classification accuracy of phoneme is at chance level. There are might be two possible reasons: 1 is the data, the other is the algorithm.
I can't find any obvious issue with the data, hence I am trying to validate the algorithm using the gesture dataset.

Result:
    -- with re-referencing: sid 10: 69%; sid 41: 97% (The good result means that the algorithm is working, so the issue should be about the data itself.);
    -- without re-referencing: sid 10: 47%; sid 41: 76%;
'''
import glob
import matplotlib
from braindecode.models import EEGNetv4, ShallowFBCSPNet

from gesture.models.deepmodel import deepnet

matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gesture.config import *
from dSPEECH.phoneme.util import train_test_split, wind_list_of_2D
from gesture.models.d2l_resnet import d2lresnet_simple, d2lresnet
from gesture.utils import windowed_data, read_data_split_function

sid=10
model_name = 'deepnet'
train_mode='original' #'DA'/original/'selected_channels'
re_referencing=False
cv=1

fs=1000
wind = 500
stride = 200
method=None
gen_data_all=None
selected_channels = False
test_epochs, val_epochs, train_epochs=read_data_split_function(sid, fs, selected_channels=selected_channels,scaler='std',cv_idx=cv,re_referencing=re_referencing)
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride,
                                                        gen_data_all=gen_data_all,train_mode=train_mode,method=method)

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

lr = 0.01
weight_decay = 1e-10
epoch_num = 500
patients=20

model_name='resnet'
class_number=5
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




