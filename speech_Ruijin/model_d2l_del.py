import torch
from torch import nn
from d2l import torch as d2l
from common_dl import device
SOS=0.1
class Seq2SeqEncoder2(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    def __init__(self, in_features, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding = nn.Linear(in_features,embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)

    def forward(self, X, *args):
        # X shape: (steps, batch_size, num_steps)
        #embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        embs=self.embedding(X)
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state

class AttentionDecoder(d2l.Decoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, out_features, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.out_features=out_features
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding = nn.Linear(out_features, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(out_features)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens=None):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state,prediction=False): # X is the target, X shape: (num_steps, batch_size, embed_size)
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)

        X=self.embedding(X)
        outputs, self._attention_weights = [torch.ones((1,X.shape[1],X.shape[2]))*SOS], [] # mark
        #x=torch.ones(X[0].shape)*SOS # TODO: how to choose SOS?
        # full teacher force
        for i,x in enumerate(X):  # k, 1,2,3,4,5 --> k,1.1,2.2,3.3,4.4,5.5,6.6
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            if i==0:
                tmp = torch.cat((context, outputs[-1].to(device).transpose(0, 1)), dim=-1)
            else:
                if not prediction: # training
                    tmp = torch.cat((context, torch.unsqueeze(X[i-1], dim=1)), dim=-1) #
                else:
                    tmp = torch.cat((context, outputs[-1].transpose(0,1)), dim=2)
                    # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(tmp.permute(1, 0, 2), hidden_state) # out: torch.Size([1, 1, 256])
            outputs.append(out) # outputs.append(x)
            self._attention_weights.append(self.attention.attention_weights)
            #x=X[i]
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size=freq_bins)
        outputs = self.dense(torch.cat(outputs[1:], dim=0)) # mark ([230, 1, 256])-->([230, 1, 80])
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

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

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)

    def predict_step(self, input, output, num_steps, save_attention_weights=True):
        """Defined in :numref:`sec_seq2seq_training`"""
        #batch = [d2l.to(a, device) for a in batch]
        #src, tgt, src_valid_len, _ = batch
        #self.inference=True
        enc_valid_len=None
        enc_all_outputs = self.encoder(input)
        dec_state = self.decoder.init_state(enc_all_outputs)
        # Return decoder output only
        return self.decoder(output, dec_state,prediction=True)

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




