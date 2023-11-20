import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from example_transformer.EEG_Transformer_seq2seq_master.lib.eeg_transformer import *
from pre_all import device
# This masking, combined with fact that the output embeddings are offset by one position,
# ensures that the predictions for position i can depend only on the known outputsat positions less than i. 
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Object for holding a batch of data with mask during training.
class Batch:
    def __init__(self, src, trg=None, pad=0, device=device,encoder_only=False):
        self.src = src.to(device)
        self.src_mask = torch.ones(src.size(0), 1, src.size(1)).to(device) # torch.Size([8, 1, 300])
        if trg is not None:
            if encoder_only:
                self.trg = trg.to(device)
                self.trg_y = trg.to(device)
            else:
                self.trg = trg[:, :-1,:].to(device)
                self.trg_y = trg[:, 1:,:].to(device) # used as 'teacher forcing' in training process
            self.trg_mask = self.make_std_mask(self.trg, pad).to(device)
            self.ntokens = self.trg_y.size(1)
    @staticmethod
    # Create a mask to hide padding and future words
    def make_std_mask(tgt, pad): 
        tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1),dtype = torch.long)
        tgt_mask = tgt_mask.type_as(tgt_mask.data) & subsequent_mask(tgt.size(1)).type_as(tgt_mask.data)
        return tgt_mask

# combine src and tgt as Batch Class
# force the first time-step to be start_symbol which indicates the start of a sequence
# TODO: force the first to start_symbol?
def data_gen(dataloader,start_symbol = 1,encoder_only=False):
    for idx, (data_x,data_y) in enumerate(dataloader): #x:(10batch, 300time, 10channel); y:(10batch, 200time, 2channel)
        if not encoder_only:
            data_x[:, 0, :] = start_symbol
            data_y[:, 0, :] = start_symbol
        src_ = data_x.float()
        tgt_ = data_y.float()
        yield Batch(src_, tgt_, 0,encoder_only=encoder_only)

# run the model and record loss
#def run_epoch(data_iter, model,criterion,optimizer):
from tqdm import tqdm
def run_epoch(data_iter, model, loss_compute):
    total_loss = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, # src:torch.Size([50, 300, 10]); trg:torch.Size([50, 299, 2])
                            batch.src_mask, batch.trg_mask) #:batch.src_mask: all 1s; batch.trg_mask:torch.Size([50, 299, 299])
        '''
        loss=criterion(out.contiguous(), batch.trg_y.contiguous())
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        '''
        loss = loss_compute(out, batch.trg_y,batch.ntokens)
        total_loss += loss
    return total_loss / (i+1), out, batch.trg # out: torch.Size([50, 199, 2])

# not suitable for encoder_only mode; just use the validation dataset to monitor the model performance (no teacher force in encoder_only mode)
def output_prediction(model,ts_test,ls_test, max_len, start_symbol,output_d,device,encoder_only=False):
    #from common_dl import device
    # we pre-process the data like in data generation
    input_data = torch.from_numpy(ts_test).unsqueeze(0).float().to(device)
    #input_data=ts_test.float()
    input_data[:,0, :] = 1
    true_output = torch.from_numpy(ls_test).unsqueeze(0).float().to(device)
    #true_output=ls_test.float()
    true_output[:,0, :] = 1
    src = input_data.detach()
    tgt = true_output.detach()
    
    test_batch = Batch(src, tgt, 0, device=device,encoder_only=encoder_only)
    src = test_batch.src
    src_mask = test_batch.src_mask
    
    model.eval()
    # feed input to encoder to get memory output which is one of the inputs to decoder
    memory = model.encode(src.float(), src_mask)
    ys = torch.ones(1, 1, output_d).fill_(start_symbol).to(device)
    
    # apply a loop to generate output sequence one by one. 
    # This means to generate the fourth output we feed the first three generated output 
    # along with memory output from encoder to decoder. 
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        out = model.generator(out)
        # concatenate new output to the previous output sequence
        ys = torch.cat([ys, out[:,[-1],:]], dim=1)
    outs = [out for out in ys]
    # return predicted sequence and true output sequence
    return torch.stack(outs,0).cpu().detach().numpy(), true_output.cpu().detach().numpy()


# not suitable for encoder_only mode; just use the validation dataset to monitor the model performance (no teacher force in encoder_only mode)
def output_prediction2(model, ts_test, ls_test, max_len, start_symbol, output_d, device, encoder_only=False):
    # from common_dl import device
    # we pre-process the data like in data generation
    #input_data = torch.from_numpy(ts_test).unsqueeze(0).float().to(device)
    # input_data=ts_test.float()
    ts_test[:, 0, :] = 1
    #true_output = torch.from_numpy(ls_test).unsqueeze(0).float().to(device)
    # true_output=ls_test.float()
    ls_test[:, 0, :] = 1
    src = ts_test.detach()
    tgt = ls_test.detach()
    batch_size=src.shape[0]

    test_batch = Batch(src, tgt, 0, device=device, encoder_only=encoder_only)
    src = test_batch.src
    src_mask = test_batch.src_mask

    with torch.no_grad():
        # feed input to encoder to get memory output which is one of the inputs to decoder
        memory = model.encode(src.float(), src_mask)
        ys = torch.ones(batch_size, 1, output_d).fill_(start_symbol).to(device)

        # apply a loop to generate output sequence one by one.
        # This means to generate the fourth output we feed the first three generated output
        # along with memory output from encoder to decoder.
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
            out = model.generator(out)
            # concatenate new output to the previous output sequence
            ys = torch.cat([ys, out[:, [-1], :]], dim=1)
            del out
        outs = [out for out in ys]
        # return predicted sequence and true output sequence
    return torch.stack(outs, 0).cpu().detach(), ls_test.cpu().detach()





