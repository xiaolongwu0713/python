'''
training mode: pretrain/resume/transfer_learning(TL);
pretrain: validate on particular one subject while test on another particular subject;train on the rest subject;
transfer learning(TL): use one particular task to fine tune the pretrained model;
'''
import copy
import sys
import socket
from datetime import datetime

from tqdm import tqdm

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import os, re
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch
from common_dl import set_random_seeds
from common_dl import myDataset
from comm_utils import slide_epochs
from torch.utils.data import DataLoader
from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da,deepnet_changeDepth,deepnet_expandPlan
from gesture.models.d2l_resnet import d2lresnet
import logging
from comm_utils import running_from_IDE, running_from_CMD
from sEMG_Zhangbin.dataset_LOO2 import sub_split, windowing, sub_split2
from sEMG_Zhangbin.config import *
import time

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

if running_from_CMD:
    testing=bool(int(sys.argv[1])) # 0/1
    class_number=int(sys.argv[2]) #  2/3/4
    cat=sys.argv[3] # binary classification: ET/PD/others/NC
    test_sub=sys.argv[4]
    val_sub=sys.argv[5]
elif running_from_IDE:
    testing = True  # True/False
    class_number = 2
    cat='ET'
    test_sub = 'TP003'
    val_sub = 'TP004'


model_name = 'resnet' #'deepnet'/resnet
lr = 0.001
resume=False
mode='pretrain' # pretrain/TL/resume

if testing==True:
    print("Testing.....")
    train_epochs = 5
    patients = 3
else:
    print("Actual Running.....")
    train_epochs = 5
    patients = 2
if resume==False:
    resume_epoch=0
else:
    resume_epoch = 4
import pytz
the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
result_folder=result_dir+the_time.strftime('%Y%m%d')
isExist = os.path.exists(result_folder)
if not isExist:
   os.makedirs(result_folder)

log_file_prefix=result_folder+'/LOO_'+mode+'_'+model_name+'_'+str(class_number)+'classes'+'_test_on_'+test_sub+'_val_on_'+val_sub+\
                '_'+the_time.strftime('%H%M%S')
log_file=log_file_prefix+'.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG)
logging.info('------------Parameter: mode:%s/model:%s/classes:%s/test_sub:%s/val_sub:%s-----------',
             mode, model_name,str(class_number),test_sub,val_sub)

if mode=='TL':
    taski=[0]
    test_sub = 'TP014'
    TL_path = result_dir + 'checkpoint_resnet_4_classes_3.pth'
    ET_data_tmp, PD_data_tmp, others_data_tmp, NC_data_tmp, test_data_tmp, val_data_tmp = sub_split(test_sub=test_sub,taski=taski)
elif mode=='resume':
    resume_path=result_dir + 'checkpoint_resnet_'+str(class_number)+'_classes_'+str(resume_epoch)+'.pth'
elif mode=='pretrain':
    #test_sub = 'TP005'
    #val_sub='TP007'
    ET_data_tmp, PD_data_tmp, others_data_tmp, NC_data_tmp, test_data_tmp,test_task_labels_tmp, val_data_tmp,val_task_labels_tmp = \
        sub_split2(test_sub=test_sub,val_sub=val_sub)

# shape: (trial_number, window_size, channels)
# training set
wind_size=1000
stride=500
ET_train = np.concatenate([windowing(triali, wind_size, stride) for triali in ET_data_tmp]).transpose(0, 2,1)  # [87 * (317, 7,500) ]
PD_train = np.concatenate([windowing(triali, wind_size, stride) for triali in PD_data_tmp]).transpose(0, 2, 1)
others_train = np.concatenate([windowing(triali, wind_size, stride) for triali in others_data_tmp]).transpose(0, 2, 1)
NC_train = np.concatenate([windowing(triali, wind_size, stride) for triali in NC_data_tmp]).transpose(0, 2, 1)
# validate set
val_dataset = np.concatenate([windowing(triali, wind_size, stride) for triali in val_data_tmp]).transpose(0, 2, 1)
# testing set
test_dataset_tmp= [windowing(triali, wind_size, stride) for triali in test_data_tmp]
test_task_labels=[]
for i,taski in enumerate(test_data_tmp):
    test_task_labels.append([test_task_labels_tmp[i]]*len(test_dataset_tmp[i]))
test_task_labels=np.concatenate(test_task_labels)
test_dataset=np.concatenate(test_dataset_tmp).transpose(0, 2, 1)
assert len(test_task_labels)==test_dataset.shape[0]
if class_number==2:
    labels=[0,0,0,1]
    if cat == 'ET':
        y_val = 0
        y_test = 0
    elif cat == 'PD':
        y_val = 0
        y_test = 0
    elif cat == 'others':
        y_val = 0
        y_test = 0
    elif cat == 'NC':
        y_val = 1
        y_test = 1
elif class_number == 3:
    labels = [0, 1, 2]
    if cat == 'ET':
        y_val = 0
        y_test = 0
    elif cat == 'PD':
        y_val = 1
        y_test = 1
    elif cat == 'others':
        y_val = 2
        y_test = 2
elif class_number==4:
    labels=[0,1,2,3]
    if cat=='ET':
        y_val=0
        y_test=0
    elif cat=='PD':
        y_val = 1
        y_test = 1
    elif cat=='others':
        y_val = 2
        y_test = 2
    elif cat=='NC':
        y_val = 3
        y_test = 3
ET_train_y=[labels[0]]*len(ET_train)
PD_train_y=[labels[1]]*len(PD_train)
others_train_y=[labels[2]]*len(others_train)
NC_train_y=[labels[3]]*len(NC_train)
if class_number==2 or class_number==4: # all classes
    X_train=np.concatenate((ET_train,PD_train,others_train,NC_train))
    y_train=ET_train_y+PD_train_y+others_train_y+NC_train_y
elif class_number==3: # no NC class
    X_train = np.concatenate((ET_train, PD_train, others_train))
    y_train = ET_train_y + PD_train_y + others_train_y
X_val=val_dataset #np.concatenate((ET_val,PD_val,others_val,NC_val))
y_val=[y_val]*X_val.shape[0] #ET_val_y+PD_val_y+others_val_y+NC_val_y
X_test=test_dataset #np.asarray(test_dataset)
y_test=[y_test]*X_test.shape[0]

# check train/val/test distribution
check_dist=False
if check_dist:
    train=X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[2]]) #(72798, 7000)
    val=X_val.reshape([X_val.shape[0],X_val.shape[1]*X_val.shape[2]])
    test=X_test.reshape([X_test.shape[0],X_test.shape[1]*X_test.shape[2]]) #(2961, 7000)

    compare=np.concatenate((train,val,test),axis=0)
    y=[0]*train.shape[0]+[1]*val.shape[0]+[2]*test.shape[0]
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca_50 = PCA(n_components=100)
    tsne_2 = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)

    pca_result = pca_50.fit_transform(compare) # (10000, 784)-->(10000, 100)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    tsne_result = tsne_2.fit_transform(pca_result) # (10000, 50)-->(10000, 2)

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.scatter(tsne_result[:train.shape[0],0],tsne_result[:train.shape[0],1])
    ax.scatter(tsne_result[train.shape[0]:train.shape[0]+val.shape[0],0],tsne_result[train.shape[0]:train.shape[0]+val.shape[0],1])
    ax.scatter(tsne_result[-test.shape[0]:,0],tsne_result[-test.shape[0]:,1])

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
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,pin_memory=False,sampler=sampler) # ,sampler=sampler
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=False)
train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190
## test the sampler
#_,yy=next(iter(train_loader))
#dict(zip(list(yy.numpy()),[list(yy.numpy()).count(i) for i in list(yy.numpy())]))
# Extract number of chans and time steps from dataset
one_window=next(iter(train_loader))[0]
n_chans = one_window.shape[1]
input_window_samples=one_window.shape[2]

if class_number==2:
    out_dim=1
elif class_number==4:
    out_dim=4

if model_name=='resnet':
    net=d2lresnet(out_dim,end_with_logsoftmax=False) # 92%
elif model_name=='deepnet':
    net = deepnet(n_chans,out_dim,wind_size) # 81%
#net = deepnet_resnet(n_chans,n_classes,input_window_samples=input_window_samples,expand=True) # 50%
#net=TSception(208)
#net=TSception(1000,n_chans,3,3,0.5)

weight_decay = 1e-10
#batch_size = 32

#img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
net.to(device).float()

if class_number==2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)

if mode=='resume':
    savepath = resume_path
    # torch.save(state, savepath)

    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer'])
elif mode=='TL':
    savepath = TL_path
    # torch.save(state, savepath)
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

train_losses=[]
train_accs=[]
val_accs=[]

for i,epoch in enumerate(range(resume_epoch+1,resume_epoch+1+train_epochs)):
    print("------ epoch " + str(epoch)+"/"+str(resume_epoch+1+train_epochs-1) +"-----")
    net.train()
    loss_epoch = 0
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # if (cuda):
        #     trainx = trainx.float().cuda()
        # else:
        #     trainx = trainx.float()
        trainx = trainx.float().to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        if class_number==2:
            preds = [0 if predi.item() <= 0 else 1 for predi in y_pred.detach()] # [ 0 or 1 ]
        else:
            preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred.squeeze(), trainy.squeeze().cuda().float()) #  2:float()/4:long()
        else:
            loss = criterion(y_pred.squeeze(), trainy.squeeze().float()) # .float()/long()

        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        if class_number==2:
            running_corrects += torch.sum(torch.from_numpy(np.asarray(preds).squeeze() == trainy.numpy().squeeze()))
        else:
            running_corrects += torch.sum(preds.cpu().squeeze() == trainy.squeeze())
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
        logging.info("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(tqdm(val_loader)):
                # if (cuda):
                #     val_x = val_x.float().cuda()
                #     # val_y = val_y.float().cuda()
                # else:
                #     val_x = val_x.float()
                #     # val_y = val_y.float()
                val_x=val_x.float().to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                if class_number == 2:
                    preds = [0 if predi.item() < 0 else 1 for predi in outputs.detach()]  # [ 0 or 1 ]
                else:
                    preds = outputs.argmax(dim=1,keepdim=True)  # Returns the indices of the maximum value of all elements in the input tensor.

                #preds = outputs.argmax(dim=1, keepdim=True)

                #running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())
                if class_number == 2:
                    running_corrects += torch.sum(torch.from_numpy(np.asarray(preds).squeeze() == val_y.numpy().squeeze()))
                else:
                    running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())

                if testing==True:
                    break
        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        logging.info("Training loss:{:.2f},Accuracy:{:.2f}; Evaluation accuracy:{:.2f}.".format(train_loss, train_acc,val_acc))
    if i==0:
        best_epoch=resume_epoch
        best_acc=val_acc
        patient=patients
        state = {
            'net': copy.deepcopy(net.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'epoch': epoch,
            # 'loss': epoch_loss
        }
    else:
        if val_acc>best_acc:
            best_epoch=epoch
            best_acc=val_acc
            patient=patients
            state = {
                'net': copy.deepcopy(net.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'epoch': epoch,
                #'loss': epoch_loss
            }

        else:
            patient=patient-1
    if patient==0:
        break
    logging.info("patients left: {:d}".format(patient))

savepath = log_file_prefix+'.pth'

if not testing:
    torch.save(state, savepath)
    logging.info('Save checkpoint to: '+savepath+'.')
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
logging.info("Testing...")
with torch.no_grad():
    running_corrects = 0
    running_errors = 0
    test_preds=[]
    test_target=[]
    for _, (test_x, test_y) in enumerate(tqdm(test_loader)): #test_loader
        test_target.append(test_y)
        # if (cuda):
        #     test_x = test_x.float().cuda()
        #     # val_y = val_y.float().cuda()
        # else:
        #     test_x = test_x.float()
        #     # val_y = val_y.float()
        test_x=test_x.float().to(device)
        test_y=test_y.float().to(device)
        outputs = net(test_x)
        if class_number == 4:
            pred_tmp = outputs.argmax(dim=1,keepdim=True)  # Returns the indices of the maximum value of all elements in the input tensor.
        elif class_number == 2:
            pred_tmp = [0 if predi.item() < 0 else 1 for predi in outputs.detach()]  # [ 0 or 1 ]
        #_, preds = torch.max(outputs, 1)
        #preds = outputs.argmax(dim=1, keepdim=True)

        #running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
        if class_number == 2:
            running_corrects += torch.sum(torch.from_numpy(np.asarray(pred_tmp).squeeze() == test_y.cpu().numpy().squeeze()))
            running_errors += np.sum(np.asarray(pred_tmp) != test_y.cpu().numpy().squeeze())
            test_preds.append(np.asarray(pred_tmp).squeeze())
        else:
            running_corrects += torch.sum(pred_tmp.cpu().squeeze() == test_y.squeeze())
            running_errors += torch.sum(pred_tmp.cpu().squeeze() != test_y.squeeze())
            test_preds.append(pred_tmp.cpu().squeeze())

        if testing == True:
            break

test_preds_final=[j for i in test_preds for j in i]
test_acc = (running_corrects.double() / len(test_preds_final)).item()
logging.info("Correct number: "+str(running_corrects))
logging.info("Error number: "+str(running_errors))
logging.info("Total test size: "+ str(len(test_preds_final)))
logging.info("Test accuracy: {:.2f}.".format(test_acc))

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc
train_result['test_preds_final']=test_preds_final
train_result['test_target']=test_target

filename=log_file_prefix+'.npy'
np.save(filename,train_result)

'''
#load
a=np.load('/Users/xiaowu/tmp/train_result_LOO_TL_test_on_TP014.npy', allow_pickle=True).item()
pred=a['test_preds_final']
target=a['test_target']
target_list=[j.item() for i in target for j in i]
pred_list=[i.item() for i in pred]
result=[target_list,pred_list]
np.save('train_result', np.array(result, dtype=object), allow_pickle=True)

data=np.load('train_result.npy',allow_pickle=True)
'''

