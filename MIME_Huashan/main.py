import sys
import socket

from gesture.utils import windowed_data

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime
import pytz
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net

from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da,deepnet_changeDepth,deepnet_expandPlan
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.deepmodel import TSception2
from MIME_Huashan.config import *
from MIME_Huashan.utils import read_data
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
if cuda:
    torch.backends.cudnn.benchmark = True

testing=False
sids=['HHFU016','017']
sid=sids[0]
task='ME' # 'ME'/'MI'
fs=1000
wind = 500
stride = 100
batch_size = 64
gen_epochs=200
class_number=4
#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

the_time=datetime.now(pytz.timezone('Asia/Shanghai'))
result_path = result_dir + 'deepLearning/' + str(wind)+'_'+str(stride)+'/'+sid + '/'+ the_time.strftime('%Y_%m_%d') + '_' + the_time.strftime('%H_%M')+'/'
print('Result dir: '+ result_path+'.')
if not os.path.exists(result_path):
    os.makedirs(result_path)
writer = SummaryWriter(result_path)

scaler='std' # 'std'/None
test_epochs, val_epochs, train_epochs, scaler=read_data(sub_name=sid,study=task,scaler=scaler) # sutdy='ME'/'MI'

# X_train.shape: (1520, 208, 500); y_train.shape: (1520, 1);
print("Windowing trial data into small chuncks....")
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride,
                                                        gen_data_all=None,retrain=None,method=None)

chn_num=X_train.shape[1]

check_data_dist=False
if check_data_dist:
    train = X_train.reshape([X_train.shape[0], X_train.shape[1] * X_train.shape[2]])  # (72798, 7000)
    val = X_val.reshape([X_val.shape[0], X_val.shape[1] * X_val.shape[2]])
    test = X_test.reshape([X_test.shape[0], X_test.shape[1] * X_test.shape[2]])  # (2961, 7000)
    compare = np.concatenate((train, val, test), axis=0)
    y = [0] * train.shape[0] + [1] * val.shape[0] + [2] * test.shape[0]
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt

    pca_n = PCA(n_components=200)
    tsne_n = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)

    pca_result = pca_n.fit_transform(compare)  # (10000, 784)-->(10000, 100)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_n.explained_variance_ratio_)))
    tsne_result = tsne_n.fit_transform(pca_result)  # (10000, 50)-->(10000, 2)

    fig, ax = plt.subplots()
    ax.scatter(tsne_result[:train.shape[0], 0], tsne_result[:train.shape[0], 1])
    ax.scatter(tsne_result[train.shape[0]:train.shape[0] + val.shape[0], 0],
               tsne_result[train.shape[0]:train.shape[0] + val.shape[0], 1])
    ax.scatter(tsne_result[-test.shape[0]:, 0], tsne_result[-test.shape[0]:, 1])

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)


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
if model_name=='eegnet':
    #print('Here')
    net = EEGNetv4(n_chans, class_number, input_window_samples=input_window_samples, final_conv_length='auto', drop_prob=0.5)
elif model_name=='shallowFBCSPnet':
    net = ShallowFBCSPNet(n_chans,class_number,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
elif model_name=='deepnet':
    net = deepnet(n_chans,class_number,wind) # 81%
elif model_name=='deepnet_changeDepth':
    net = deepnet_changeDepth(n_chans,class_number,wind,depth) # 81%
    model_name='deepnet_changeDepth_'+str(depth)
elif model_name == 'deepnet2':
    net = deepnet_seq(n_chans, class_number, wind)
elif model_name == 'deepnet_rnn':
    net = deepnet_rnn(n_chans, class_number, wind)  # 65%
elif model_name=='resnet':
    net=d2lresnet(class_num=class_number,task='classification',end_with_logsoftmax=False) # 92%
elif model_name=='tsception':
    net = TSception2(1000, n_chans, 3, 3, 0.5)
elif model_name=='deepnet_da':
    net = deepnet_da(n_chans, class_number, wind)

lr = 0.01
weight_decay = 1e-10
epoch_num = 500
patients=20

img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
net=net.to(device)

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
    print("------ epoch" + str(epoch) + ": sid"+str(sid)+"@"+model_name+"-----")
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(train_loader):
        optimizer.zero_grad()

        trainx = trainx.float().to(device)
        trainy = trainy.type(torch.LongTensor).to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)
        loss = criterion(y_pred, trainy.squeeze())

        if model_name != 'resnet':
            for i, layer in enumerate(net.layers):
                reg_variable = reg_variable+torch.sum(torch.pow(layer.weight.detach(), 2))
            reg_variable = weight_decay * reg_variable
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.cpu().squeeze())

        if testing:
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
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
                val_x = val_x.float().to(device)
                val_y = val_y.type(torch.LongTensor).to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.cpu().squeeze())

                if testing:
                    break
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

    if testing:
        break
checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.float().to(device)
        test_y = test_y.to(device)
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)

        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.cpu().squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

filename=result_path +  model_name + '_'+ str(wind)+'_'+str(stride) + '.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename+'.npy',allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])

