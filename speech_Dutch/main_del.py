import sys
import socket

from speech_Dutch.model import Seq2SeqEncoder2, Seq2SeqAttentionDecoder, EncoderDecoder

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import numpy as np
from speech_Dutch.config import *
from speech_Dutch.utils import TacotronSTFT
from common_dl import *
from speech_Dutch.dataset import dataset
from torch.utils.data import DataLoader

sid=1
session=0
ds_train=dataset(sid,session,split='train')
ds_val=dataset(sid,session,split='val')
ds_test=dataset(sid,session,split='test')
batch_size=32
train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=ds_val, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False, pin_memory=False)

in_features=7
out_features=80
batch_size=3
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
source_len=1000
tgt_len=500
encoder = Seq2SeqEncoder2(in_features, embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(out_features, tgt_len, embed_size, num_hiddens, num_layers, dropout)
model = EncoderDecoder(encoder, decoder)

x=torch.rand(source_len,batch_size,in_features)
y=torch.rand(tgt_len,batch_size,out_features)

out=model(x,y)
pred=model.predict_step(x,500)

