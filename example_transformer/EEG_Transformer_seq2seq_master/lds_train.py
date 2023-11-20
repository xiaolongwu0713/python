from example_transformer.EEG_Transformer_seq2seq_master.lib.train import *
from pylds.models import DefaultLDS
# if error happens, change this: "from scipy.misc import logsumexp" to "from scipy.special import logsumexp"

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
npr.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)

TIMESTEPS = 300 # number of steps in time
INSTS = 500 # batch-size or the number of instances
DOBS = 10 # number of observable variables
DLAT = 2 # number of hidden variabkes (latent states)

def simple_lds(d_observed=DOBS,d_latent=DLAT,d_input=-1,timesteps=TIMESTEPS,insts=INSTS):
    ## d_observed : dimensionality of observed data
    ## d_latent : dimensionality of latent states
    ## d_input : dimensionality of input data. For d_input=-1 a model with no input is generated
    ## timesteps: number of simulated timesteps
    ## insts: number of instances
    ## instantiating an lds with a random rotational dynamics matrix

    if d_input == -1 :
        lds_model = DefaultLDS(d_observed,d_latent,0)
        input_data = None
    else:
        lds_model = DefaultLDS(d_observed,d_latent,d_input)
        input_data = npr.randn(insts,timesteps,d_input)

    # initializing the output matrices:
    training_set = np.zeros((insts, timesteps, d_observed))
    latent_states= np.zeros((insts, timesteps, d_latent))

    # running the model and generating data
    for i in range(insts):
        training_set[i,:,:], latent_states[i,:,:] = lds_model.generate(timesteps, inputs=input_data)
    return training_set, latent_states, lds_model

# Instantiating a Model and Generating Data
ts,ls,mdl = simple_lds()
ls=ls[:,:200,:] #ts:(10batch, 300time, 10channel); ls:(10batch, 200time, 2channel)
# Get input_d, output_d, timesteps from the initial dataset
input_d, output_d = ts.shape[2], ls.shape[2]
timesteps = ts.shape[1]
print('input_d:',input_d,'output_d:',output_d,'timesteps:',timesteps)

class LDSDataset(Dataset):
    # use boolen value to indicate that the data is for training or testing
    def __init__(self,x,y,train,ratio):
        self.len = x.shape[0]
        self.ratio = ratio
        split = int(self.len*self.ratio)
        self.x_train = torch.from_numpy(x[:split])
        self.y_train = torch.from_numpy(y[:split])
        self.x_test = torch.from_numpy(x[split:])
        self.y_test = torch.from_numpy(y[split:])
        self.train = train

    def __len__(self):
        if self.train:
            return int(self.len*self.ratio)
        else:
            return int(self.len*(1-self.ratio))

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]

# split training and testing set
split_ratio = 0.8
batch_size = 50
dataset_train = LDSDataset(ts,ls,True,split_ratio)
dataloader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
dataset_test = LDSDataset(ts,ls,False,split_ratio)
dataloader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=True)

opt = {}
opt['Transformer-layers'] = 2
opt['Model-dimensions'] = 256
opt['feedford-size'] = 512
opt['headers'] = 8
opt['dropout'] = 0.1
opt['src_d'] = input_d # input dimension
opt['tgt_d'] = output_d # output dimension
opt['timesteps'] = timesteps

from pre_all import device
criterion = nn.MSELoss() # mean squared error
# setup model using hyperparameters defined above
model = make_model(opt['src_d'],opt['tgt_d'],opt['Transformer-layers'],
                   opt['Model-dimensions'],opt['feedford-size'],opt['headers'],opt['dropout']).to(device)

# setup optimization function
model_opt = NoamOpt(model_size=opt['Model-dimensions'], factor=1, warmup=400,
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015, betas=(0.9, 0.98), eps=1e-9))
total_epoch = 2000
train_losses = np.zeros(total_epoch)
test_losses = np.zeros(total_epoch)

patients=30
for epoch in range(total_epoch):
    model.train()
    train_loss,out = run_epoch(data_gen(dataloader_train), model,SimpleLossCompute(model.generator, criterion, model_opt))

    train_losses[epoch]=train_loss

    if (epoch+1)%50 == 0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }, 'model_checkpoint/'+str(epoch)+'.pth')
        #torch.save(model, 'model_save/model%d.pth'%(epoch)) # save the model

    model.eval() # test the model
    test_loss = run_epoch(data_gen(dataloader_test), model,
            SimpleLossCompute(model.generator, criterion, None))
    test_losses[epoch] = test_loss
    print('Epoch[{}/{}], train_loss: {:.6f},test_loss: {:.6f}'.format(epoch+1, total_epoch, train_loss, test_loss))

    if epoch==0:
        best_test_loss=test_loss
        patient=patients
        best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }
    else:
        if test_loss>best_test_loss:
            patient=patient-1
        else:
            patient=patients
            best_model={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }
    if patient==0:
        break

# choose a pair of data from test dataset
# transfer from tensor to numpy array
test_x, test_y = dataset_test.x_test[1].numpy(),dataset_test.y_test[1].numpy()
# load best model
model.load_state_dict(best_model['model_state_dict'])
# make a prediction then compare it with its true output
test_out, true_out = output_prediction(model,test_x, test_y, max_len=opt['timesteps'], start_symbol=1,output_d=opt['tgt_d'])