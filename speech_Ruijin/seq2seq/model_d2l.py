import torch
from torch import nn
from d2l import torch as d2l

from common_dl import device

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

SOS=1
class Seq2SeqEncoder2(nn.Module):
    """The RNN encoder for sequence to sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    def __init__(self, in_features, embed_size, num_hiddens, num_layers,embedding=True,
                 dropout=0,bidirectional=False,ts_conv=True):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.ts_conv=ts_conv # temporal and spatial convolution
        self.bidirectional=bidirectional # not implemented
        self.num_layers=num_layers
        self.embedding=embedding
        if self.embedding:
            self.embedding_layer = nn.Linear(in_features,embed_size)
            self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(in_features, num_hiddens, num_layers, dropout=dropout)

        self.Tception = nn.Sequential(
            nn.Conv2d(1, in_features, kernel_size=(1, 101), stride=1, padding=(0, 50)),  # kernel: 500
            nn.BatchNorm2d(in_features),
            nn.ReLU())

        self.Sception = nn.Sequential(
            nn.Conv2d(1, embed_size, kernel_size=(in_features, 1), stride=1, padding=0),
            nn.BatchNorm2d(embed_size),
            nn.ReLU())

    def forward(self, X, *args):
        # X shape: #(time,batch,feature) torch.Size([20, 3, 127])
        #embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        if self.embedding:
            if self.ts_conv:
                tmp=X[None,:,:,:].permute(2,0,3,1) # (batch, plan, channle, len) torch.Size([3, 1, 127, 20])
                tmp=self.Sception(tmp) # torch.Size([3, 256, 1, 20])
                embs=tmp.squeeze(2).permute(2,0,1) # torch.Size([20, 3, 256])
            else:
                embs=self.embedding_layer(X) # ([430, 3, 256])
            # embs shape: (num_steps, batch_size, embed_size)
            outputs, state = self.rnn(embs)
            # outputs shape: (num_steps, batch_size, num_hiddens)
            # state shape: (num_layers, batch_size, num_hiddens)
            #tmp=state.view(self.num_layers,2,state.shape[-2],state.shape[-1])
        else:
            outputs, state = self.rnn(X)
        return outputs, state # outputs/state: torch.Size([2, 3, 256])

class AttentionDecoder(nn.Module):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, out_features, embed_size, num_hiddens, num_layers,embedding=True,batch_size=32,dropout=0):
        super().__init__()
        self.batch_size=batch_size
        self.out_features=out_features
        self.embedding=embedding
        self.embed_size=embed_size
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_layer = nn.Linear(out_features, embed_size)
        if self.embedding:
            self.rnn = nn.GRU(
                embed_size + num_hiddens, num_hiddens, num_layers,
                dropout=dropout)
        else:
            self.rnn = nn.GRU(
                out_features + num_hiddens, num_hiddens, num_layers,
                dropout=dropout)
        self.dense = nn.LazyLinear(out_features)
        self.apply(d2l.init_seq2seq)
        # TODO: too many linear might smooth the data, hence lower frequency resolution?
        #self.last_linear_layers=torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.embed_size), torch.nn.ELU(),torch.nn.Dropout(0.5), torch.nn.Linear(self.embed_size, self.out_features))
        self.last_linear_layers=torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.out_features))
    def init_state(self, enc_outputs, enc_valid_lens=None):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self,state, num_steps, target=None,prediction=False,use_pre_prediction=True): # target is the target, target shape: (num_steps, batch_size, embed_size)
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output target: (num_steps, batch_size, embed_size)
        #if self.embedding:
        #    target=self.embedding_layer(target)
        embedded_pre_outputs, outputs, self._attention_weights = [], [], [] # mark
        inp=self.embedding_layer(torch.ones((1,enc_outputs.shape[0],self.out_features)).to(device))*SOS
        embedded_pre_outputs.append(inp)
        #target=torch.ones(target[0].shape)*SOS # TODO: how to choose SOS?
        # full teacher force
        for i in range(num_steps):  # k, 1,2,3,4,5 --> k,1.1,2.2,3.3,4.4,5.5,6.6

            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of contet: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            embedded_pre_outputs2=torch.cat(embedded_pre_outputs).transpose(0,1) # (B,t,f)
            tmp1=torch.cat((context.repeat(1,embedded_pre_outputs2.shape[1],1),embedded_pre_outputs2),dim=-1)
            tmp2 = torch.cat((context, inp.transpose(0, 1)), dim=-1) # torch.Size([3, 1, 512]) # torch.Size([3, 1, 512])
            tmp=tmp1 if use_pre_prediction else tmp2
            out, hidden_state = self.rnn(tmp.permute(1, 0, 2), hidden_state) # torch.Size([1, 3, 256])  # out: torch.Size([1, 1, 256])
            out=out[[-1],:,:] # [-1]: without loss the first dimension

            linear_out=self.last_linear_layers(out) # torch.Size([1, 3, 80])
            if target is not None:# teacher force
                if 1==1:
                    teacher_factor=0.5
                    teacher_forcing = torch.bernoulli(target.new_ones((target.shape[1],)) * teacher_factor).byte()
                    forced = torch.where(teacher_forcing.repeat(target.shape[2], 1).T, target[i,:, :], linear_out.squeeze(0)).unsqueeze(0)
                    linear_out=forced
            inp=self.embedding_layer(linear_out)  #torch.Size([1, 3, 256])
            outputs.append(linear_out)
            embedded_pre_outputs.append(inp)
            self._attention_weights.append(self.attention.attention_weights)


        outputs = torch.cat(outputs, dim=0)
        return outputs, [enc_outputs, hidden_state, enc_valid_lens],self._attention_weights

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(torch.nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_features=self.decoder.out_features

    def forward(self, enc_X, dec_X, *args): # enc_X:(time,batch,channel); dec_X:(time, batch, feature/mel_bins)
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        output,state,weights=self.decoder(dec_state,dec_X.shape[0], dec_X)
        return output,state

    def predict_step(self, input, output, num_steps, save_attention_weights=True):
        """Defined in :numref:`sec_seq2seq_training`"""
        #batch = [d2l.to(a, device) for a in batch]
        #src, tgt, src_valid_len, _ = batch
        #self.inference=True
        enc_valid_len=None
        enc_all_outputs = self.encoder(input)
        dec_state = self.decoder.init_state(enc_all_outputs)
        # no teacher force during decoding
        output,state,weights=self.decoder(dec_state,output.shape[0])
        return output,state,weights

        '''
        dec_state = self.decoder.init_state(enc_all_outputs, enc_valid_len)
        # shape of output: (num_steps, batch_size, embed_size)
        # SOS = ones
        batch_size=input.shape[1]
        outputs, attention_weights = [torch.ones(1,batch_size,self.out_features)*SOS, ], []
        for _ in range(num_steps): # k, 1,2,3,4,5 --> k,1.1,2.2,3.3,4.4,5.5,6.6
            enc_outputs, hidden_state, enc_valid_lens = dec_state
            X = self.decoder.embedding(outputs[-1])
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.decoder.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(X[0], dim=1)), dim=-1)
            out, hidden_state = self.decoder.rnn(x.permute(1, 0, 2), hidden_state)  # out: torch.Size([1, 1, 256])

            out = self.decoder.dense(out)  # mark ([230, 1, 256])-->([230, 1, 80])
            outputs.append(out)
            attention_weights.append(self.decoder.attention.attention_weights)
            #Y, dec_state = self.decoder(outputs[-1], dec_state) # Y:(num_steps, batch_size, vocab_size=freq_bins)
            #outputs.append(Y)
            #outputs.append(d2l.argmax(Y, 2)) # TODO: not the argmax in regression task
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        #return d2l.concat(outputs[1:], 0), attention_weights
        return d2l.concat(outputs[:-1], 0), attention_weights #mark
        '''


def test_seq2seq_model(net,dataloader_test):
    loss_fun = torch.nn.MSELoss()
    predictions=[]
    truths=[]
    net.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(dataloader_test):
            #src = test_x.float().permute(1,0,2).to(device) # (batch,time,feature)-->#(time,batch,feature)
            #tgt = test_y.float().permute(1,0,2).to(device)
            src = test_x.float().permute(2,0,1).to(device)  # (batch,time,feature)-->#(time,batch,feature)
            tgt = test_y.float().permute(2,0,1).to(device)
            # src,tgt=src.permute(1,0,2),tgt.permute(1,0,2) # ECoG_seq2seq
            out_len = tgt.shape[0]
            # no teacher force during validation
            output, _, attention_weights = net.predict_step(src, tgt[1:, :, :], out_len)  # torch.Size([1, 194, 80])
            loss = loss_fun(output, tgt[1:, :, :])
            predictions.append(output.cpu().numpy())
            truths.append(tgt[1:, :, :].cpu().numpy())
    predictions2 = np.concatenate(predictions, axis=1).transpose(1,0,2)
    truths2 = np.concatenate(truths, axis=1).transpose(1,0,2)

    #filename1 = result_dir + 'predictions.npy'
    #np.save(filename1, predictions2)
    #filename2 = result_dir + 'truths.npy'
    #np.save(filename2, truths2)

    return predictions2,truths2



