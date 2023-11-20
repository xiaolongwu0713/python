'''
Transfer learning should use the save train/val/test partitioning;

BIG ISSUE：
The particular task data is already been used in the previous training. This is not the same case as the transfer learning
from other's work.
'''
import torch
import numpy as np
import random,math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

from common_dl import set_random_seeds, myDataset
from gesture.models.d2l_resnet import d2lresnet
from sEMG_Zhangbin.dataset_LOO2 import sub_split, windowing
from sEMG_Zhangbin.config import *

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

wind_size=1000
stride=200
class_number=4
testing=False # True/False
resume=False
if testing==True:
    train_epochs = 1
    patients = 3
else:
    train_epochs = 5
    patients = 3
if resume==False:
    resume_epoch=0
else:
    resume_epoch = 4

model_name='resnet'
test_sub='TP014'
ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp,test_data_tmp,val_data_tmp=sub_split(taski=[0],test_sub=test_sub)

ET_train = np.concatenate([windowing(triali, wind_size, stride) for triali in ET_data_tmp]).transpose(0, 2,1)  # [87 * (317, 7,500) ]
PD_train = np.concatenate([windowing(triali, wind_size, stride) for triali in PD_data_tmp]).transpose(0, 2, 1)
others_train = np.concatenate([windowing(triali, wind_size, stride) for triali in others_data_tmp]).transpose(0, 2, 1)
NC_train = np.concatenate([windowing(triali, wind_size, stride) for triali in NC_data_tmp]).transpose(0, 2, 1)
# validate set
val_dataset = np.concatenate([windowing(triali, wind_size, stride) for triali in val_data_tmp]).transpose(0, 2, 1)
# testing set
test_dataset = np.concatenate([windowing(triali, wind_size, stride) for triali in test_data_tmp]).transpose(0, 2, 1)

if class_number==2:
    labels=[0,0,0,1]
elif class_number==4:
    labels=[0,1,2,3]

ET_train_y=[labels[0]]*len(ET_train)
PD_train_y=[labels[1]]*len(PD_train)
others_train_y=[labels[2]]*len(others_train)
NC_train_y=[labels[3]]*len(NC_train)

X_train=np.concatenate((ET_train,PD_train,others_train,NC_train))
y_train=ET_train_y+PD_train_y+others_train_y+NC_train_y
X_val=val_dataset #np.concatenate((ET_val,PD_val,others_val,NC_val))
y_val=[1]*X_val.shape[0] #ET_val_y+PD_val_y+others_val_y+NC_val_y
X_test=test_dataset #np.asarray(test_dataset)
y_test=[1]*X_test.shape[0]


uni_labels=sorted(set(labels))
lab_num=[y_train.count(labi) for labi in uni_labels]
weights=[sum(lab_num)/numi for numi in lab_num]
#weights=[X_train.shape[0] / datai.shape[0] for datai in [ET_train, PD_train, others_train, NC_train]]
#weight = 1. / class_sample_count
samples_weight = np.array([weights[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_set=myDataset(X_train,np.asarray(y_train)) #len(train_set):217809
val_set=myDataset(X_val,np.asarray(y_val)) #len(val_set):72603
test_set=myDataset(X_test,np.asarray(y_test)) #len(test_set):72606

batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,pin_memory=False,sampler=sampler)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190


out_dim=4
lr=0.001
net=d2lresnet(out_dim,end_with_logsoftmax=False) # 92%
net=net.to(device)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
if class_number==2:
    criterion = torch.nn.BCEWithLogitsLoss()
elif class_number==4:
    criterion = torch.nn.CrossEntropyLoss()

savepath = result_dir+'checkpoint_resnet_4_classes_3.pth'
checkpoint = torch.load(savepath,map_location=device)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])


train_losses=[]
train_accs=[]
val_accs=[]
resume_epoch=0
train_epochs=10
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for i,epoch in enumerate(range(resume_epoch+1,resume_epoch+1+train_epochs)):
    print("------ epoch " + str(epoch)+"/"+str(resume_epoch+1+train_epochs-1) +"-----")
    net.train()
    loss_epoch = 0
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
        else:
            trainx = trainx.float()
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        if class_number==4:
            preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        elif class_number==2:
            preds = [0 if predi.item() <= 0 else 1 for predi in y_pred.detach()] # [ 0 or 1 ]
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred.squeeze(), trainy.squeeze().cuda().long()) #  2:float()/4:long()
        else:
            loss = criterion(y_pred.squeeze(), trainy.squeeze().long()) # .float()

        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        if class_number==4:
            running_corrects += torch.sum(preds.cpu().squeeze() == trainy.squeeze())
        elif class_number==2:
            running_corrects += torch.sum(torch.from_numpy(np.asarray(preds).squeeze() == trainy.numpy().squeeze()))
        if testing==True:
            break
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
        print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(tqdm(val_loader)):
                if (cuda):
                    val_x = val_x.float().cuda()
                    # val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    # val_y = val_y.float()
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                if class_number == 4:
                    preds = outputs.argmax(dim=1,keepdim=True)  # Returns the indices of the maximum value of all elements in the input tensor.
                elif class_number == 2:
                    preds = [0 if predi.item() < 0 else 1 for predi in outputs.detach()]  # [ 0 or 1 ]
                #preds = outputs.argmax(dim=1, keepdim=True)

                #running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())
                if class_number == 4:
                    running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())
                elif class_number == 2:
                    running_corrects += torch.sum(torch.from_numpy(np.asarray(preds).squeeze() == val_y.numpy().squeeze()))
                if testing==True:
                    break
        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        print("Training loss:{:.2f},Accuracy:{:.2f}; Evaluation accuracy:{:.2f}.".format(train_loss, train_acc,val_acc))
    if i==0:
        best_epoch=resume_epoch
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
    if patient==0:
        break
    print("patients left: {:d}".format(patient))

savepath = result_dir + 'checkpoint_TL_'+model_name+'_' +str(class_number)+'classes_epoch'+ str(best_epoch)\
           +'_test_on_'+test_sub+ '.pth'

if not testing:
    torch.save(state, savepath)
    print('Save checkpoint to: '+savepath+'.')
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#net.eval()
print("Testing...")
#with torch.no_grad():
#pred_fi=[]
#target_fi=[]
running_corrects = 0
running_errors = 0
test_preds=[]
test_target=[]
for _, (test_x, test_y) in enumerate(tqdm(test_loader)):
    test_target.append(test_y)
    if (cuda):
        test_x = test_x.float().cuda()
        # val_y = val_y.float().cuda()
    else:
        test_x = test_x.float()
        # val_y = val_y.float()
    outputs = net(test_x)
    if class_number == 4:
        pred_tmp = outputs.argmax(dim=1,keepdim=True)  # Returns the indices of the maximum value of all elements in the input tensor.
    elif class_number == 2:
        pred_tmp = [0 if predi.item() < 0 else 1 for predi in outputs.detach()]  # [ 0 or 1 ]
    #_, preds = torch.max(outputs, 1)
    #preds = outputs.argmax(dim=1, keepdim=True)

    #running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
    if class_number == 4:
        running_corrects += torch.sum(pred_tmp.cpu().squeeze() == test_y.squeeze())
        running_errors += torch.sum(pred_tmp.cpu().squeeze() != test_y.squeeze())
        test_preds.append(pred_tmp.cpu().squeeze())
    elif class_number == 2:
        running_corrects += torch.sum(torch.from_numpy(np.asarray(pred_tmp).squeeze() == test_y.numpy().squeeze()))
        test_preds.append(np.asarray(pred_tmp).squeeze())
    if testing == True:
        break

test_preds_final=[j for i in test_preds for j in i]
test_acc = (running_corrects.double() / len(test_preds_final)).item()
print("Correct number: "+str(running_corrects))
print("Error number: "+str(running_errors))
print("Total test size: "+ str(len(test_preds_final)))
print("Test accuracy: {:.2f}.".format(test_acc))

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc
train_result['test_preds_final']=test_preds_final
train_result['test_target']=test_target

filename=result_dir + 'train_result_TL_LOO_test_on_'+test_sub+'.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename+'.npy',allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])
