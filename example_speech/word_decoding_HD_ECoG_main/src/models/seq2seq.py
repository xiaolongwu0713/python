'''
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, drop_ratio=.0, n_enc_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_enc_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=n_enc_layers,
                          bidirectional=bidirectional)

    def forward(self, input):
        output, hidden = self.gru(input)
        if self.gru.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output,hidden


class Decoder(nn.Module):
    def __init__(self, output_size, drop_ratio=.0, n_dec_layers=1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.num_layers = n_dec_layers
        self.gru = nn.GRU(output_size, output_size,
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=self.num_layers)

    def forward(self, encoder_outputs, out_len):

        # last encoder output is the first hidden
        hidden = encoder_outputs[:, :, -1]
        input = self.initInput(hidden.shape[1], hidden.device)
        output = []
        for i in range(out_len):
            temp, hidden = self.gru(input, hidden)
            input = temp.detach()
            output.append(temp)
        return torch.cat(output)

    def initInput(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.output_size, device=device)

# Bahdanau attention, also called additive attention:
# Additive Attention: http://www.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size)) # why batch-dependant?
        self.va = nn.Parameter(torch.FloatTensor(hidden_size))
        torch.nn.init.normal_(self.va) # otherwise init huge values

    def forward(self, last_hidden, encoder_outputs):
        batch_size, seq_lens, _ = encoder_outputs.size()

        x = last_hidden.transpose(0, 1) # torch.Size([32, 1, 14])
        out = torch.tanh(self.Wa(x).sum(1, keepdim=True) + self.Ua(encoder_outputs))
        attention_score=out.matmul(self.va)
        attention_weight=F.softmax(attention_score, -1)
        #attention_energies = self._score(last_hidden, encoder_outputs)

        return attention_weight

    def _score(self, last_hidden, encoder_outputs):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :return: a score (batch_size, max_time)
        """
        x = last_hidden.transpose(0, 1)
        # TODO: why sum here?
        out = torch.tanh(self.Wa(x).sum(1, keepdim=True) + self.Ua(encoder_outputs))
        #return out.bmm(self.va.unsqueeze(2)).squeeze(-1)
        return out.matmul(self.va)


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, drop_ratio=.0, n_dec_layers=1):
        super(AttentionDecoder, self).__init__()
        self.num_layers = n_dec_layers
        self.output_size = output_size
        self.gru = nn.GRU(2*output_size, output_size, # the 2*output_size comes from: concat_input = torch.cat((input, context), 2)
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=self.num_layers)
        self.attn = Attention(output_size)

    def initInput(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.output_size, device=device)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.output_size, device=device)

    def forward(self, encoder_outputs, encoder_hidden, y, out_len,teacher_force=True):
        init_with_encoder=True
        input = self.initInput(encoder_outputs.shape[0], encoder_outputs.device) # torch.Size([100, 1, 80])
        if not init_with_encoder:
            hidden = self.initHidden(encoder_outputs.shape[0], encoder_outputs.device)
        else:
            #input = torch.zeros(encoder_outputs.shape[0], 1, self.output_size, device=encoder_outputs.device) # torch.Size([32, 1, 14])
            hidden=encoder_hidden # torch.Size([1, 32, 14])

        output = []
        for i in range(y.shape[1]): # 1 2 3 4 5;
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(hidden, encoder_outputs)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x N
            concat_input = torch.cat((input, context), 2)
            temp, hidden = self.gru(concat_input, hidden)
            if teacher_force:
                # TODO: introduce a teacher force propability??
                input = y[:, i, :].unsqueeze(1)  # teacher force
            else:
                input = temp.detach()
            output.append(temp)
        #return output, attn_weights
        return torch.cat(output,1), attn_weights


class Seq2seq(nn.Module):
    def __init__(self, input_size, output_size,out_lan,
                 drop_ratio=0,
                 n_enc_layers=1,
                 n_dec_layers=1,
                 enc_bidirectional=False):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(input_size, output_size, drop_ratio, n_enc_layers, enc_bidirectional)
        #self.decoder = Decoder(output_size)
        self.decoder = AttentionDecoder(output_size, drop_ratio, n_dec_layers)
        self.out_lan=out_lan

    def forward(self, x, y):
        x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)  # ECoG_seq2seq
        encoder_outputs, encoder_hidden = self.encoder(x)  # x should be Batch x Length x Channels
        dec_output, attn_weights = self.decoder(encoder_outputs, encoder_hidden, y, self.out_lan)
        return dec_output.permute(1,0,2), attn_weights

    def predict_step(self,x,y):
        x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)  # ECoG_seq2seq
        encoder_outputs, encoder_hidden = self.encoder(x)  # x should be Batch x Length x Channels
        dec_output, attn_weights = self.decoder(encoder_outputs, encoder_hidden, y, self.out_lan,teacher_force=False)
        return dec_output.permute(1,0,2), attn_weights


    def get_attn_weights(self, x, out_len=1):
        _, attn_weights = self._forward(x, out_len)
        return attn_weights

def make_model(in_dim, out_dim, out_len, **kwargs):
    model = Seq2seq(in_dim, out_dim, out_len, **kwargs)
    return model

'''
n_in=118
n_out=80
out_len=500
drop_ratio=0.5
seq_n_enc_layers=1
seq_n_dec_layers=1
seq_bidirectional=False
device = torch.device("cpu")
net = make_model(in_dim=n_in,
                 out_dim=n_out,
                 out_len=out_len,
                 drop_ratio=drop_ratio,
                 n_enc_layers=seq_n_enc_layers,
                 n_dec_layers=seq_n_dec_layers,
                 enc_bidirectional=seq_bidirectional).to(device)

x=torch.rand(32,1000,n_in)
y=torch.rand(32,out_len,n_out)
out=net(x,y) # torch.Size([32, 14, 1])

'''


