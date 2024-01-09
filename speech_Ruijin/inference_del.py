import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from common_dl import device
from speech_Ruijin.seq2seq.dataset_seq2seq import dataset_seq2seq
from speech_Ruijin.model import Seq2SeqEncoder2, Seq2SeqAttentionDecoder, EncoderDecoder

sid=1
window_size=400
batch_size=9

train_ds,val_ds,test_ds=dataset_seq2seq(sid,window_size=window_size)
train_size=len(train_ds)
val_size=len(val_ds)
test_size=len(test_ds)

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

ch_num=train_ds[0][0].shape[0]
in_features=ch_num
out_features=80
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
#source_len=1000
#tgt_len=500
encoder = Seq2SeqEncoder2(in_features, embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(out_features, embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder).to(device)

loss_fun=torch.nn.MSELoss()
learning_rate=0.0001
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

filename='/Users/xiaowu/My Drive/tmp/model/checkpoint_49.pth'
checkpoint = torch.load(filename)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

x,y=next(iter(train_loader))
src = x.float().permute(2,0,1).to(device)
tgt = y.float().permute(2,0,1).to(device)
y_pred = net.predict_step(src,76) # torch.Size([76, 9, 80])

pred=y_pred[0].detach().numpy()
src=src.numpy()
fig,ax=plt.subplots()
ax.imshow(pred[:,0,:])
ax.clear()
ax.imshow(src[:,2,:].transpose())
pred[0,0,:]



