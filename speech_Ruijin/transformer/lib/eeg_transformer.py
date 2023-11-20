import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

extract_attention=True

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_d),c(position)),
        nn.Sequential(Embeddings(d_model, tgt_d),c(position)),
        Generator(d_model,tgt_d))
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,encoder_only=False,
                 spatial_attention_layer=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.encoder_only=encoder_only
        self.spatial_attention_encoder=spatial_attention_layer
        self.emb_spatial_attention1 = Embeddings(256, 151)
        self.emb_spatial_attention2 = Embeddings(151, 256)
    # src: torch.Size([128, 151 time, 127 channel]); tgt:torch.Size([128, 99 time, 40 melbins]); src_mask:torch.Size([128, 1, 151])
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #print('forward')
        if self.spatial_attention_encoder is not None:
            src=self.spatial_attention_encode(src.transpose(1,2),torch.ones(src.shape[0],1,127).cuda())
            src=src.transpose(1,2)
        if self.encoder_only:
            return self.generator(self.encode(src, src_mask))
        if extract_attention:
            # att_encoder: list of N (encoder layer number) matrix: N * torch.Size([64, 8, 151, 151])
            tmp,att_encoder=self.encode(src, src_mask) # att_encoder:6*[(8, 151, 151)]
            # att_decoder(N* torch.Size([8, 99, 99]))/att_enc_dec(N*torch.Size([8, 99, 151])): list of N (decoder layer number) matrix
            tmp, att_decoder, att_enc_dec = self.decode(tmp, src_mask, tgt, tgt_mask)
            return self.generator(tmp), att_encoder, att_decoder, att_enc_dec
        else:
            return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def spatial_attention_encode(self,src,src_mask):
        src_tmp,att=self.spatial_attention_encoder(self.emb_spatial_attention1(src), src_mask)
        return self.emb_spatial_attention2(src_tmp)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # decoder: forward(self, x, memory, src_mask, tgt_mask)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear"
    def __init__(self, d_model,output_d,norm_mel=False):
        super(Generator, self).__init__()
        if norm_mel:
            self.proj1 = nn.Sequential(nn.Linear(d_model, output_d), )
            #self.proj1 = nn.Sequential(nn.Linear(d_model, output_d), nn.Tanh())
        else:
            self.proj1 = nn.Sequential(nn.Linear(d_model, output_d),)
        self.norm_mel=norm_mel

    def forward(self, x):
        return self.proj1(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        atts = []
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x,att = layer(x, mask) # att: (8, 151, 151)
            atts.append(att)
        return self.norm(x),atts # self.att: 6*[(8, 151, 151)]


class test(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self):
        super(test, self).__init__()
        self.l=nn.Linear(4,5)
        self.att = []

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x=self.l(x)
        self.att.append(x)
        return x,self.att

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # self.sublayer is the residual connection wrapper
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        if not extract_attention:
            "Follow Figure 1 (left) for connections."
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # torch.Size([50, 300, 256])
        else:
            # unfold the sublayer connection function
            # take out the wrapped function (fun), then separate the self.sublayer[0] into norm and dropout opts.
            func=lambda x: self.self_attn(x, x, x, mask)
            tmp,attention = func(self.sublayer[0].norm(x))
            x = x + self.sublayer[0].dropout(tmp)
        return self.sublayer[1](x, self.feed_forward),attention

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    # x is the target matrix
    def forward(self, x, memory, src_mask, tgt_mask):
        att_decoders = []
        att_enc_decs = []
        if extract_attention:
            for layer in self.layers:
                x, att_decoder, att_enc_dec = layer(x, memory, src_mask, tgt_mask)
                att_decoders.append(att_decoder)
                att_enc_decs.append(att_enc_dec)
            return self.norm(x),att_decoders,att_enc_decs
        else:
            for layer in self.layers:
                x, _, _ = layer(x, memory, src_mask, tgt_mask)
            return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        if extract_attention:
            # unfold the sublayer residual connection
            func1=lambda x: self.self_attn(x, x, x, tgt_mask)
            func2=lambda x: self.src_attn(x, m, m, src_mask)
            tmp,att_decoder=func1(self.sublayer[0].norm(x))
            x = x + self.sublayer[0].dropout(tmp)

            tmp,att_enc_dec=func2(self.sublayer[1].norm(x))
            x = x + self.sublayer[1].dropout(tmp)

            return self.sublayer[2](x, self.feed_forward), att_decoder, att_enc_dec

        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None: # mask: torch.Size([120, 1, 1, 20])]]
        scores = scores.masked_fill(mask == 0, -1e9) # ???
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # TODO: should be 3 instead of 4?
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # torch.Size([50, 1, 1, 300]): all 1s
        nbatches = query.size(0)
        # query/key/value: torch.Size([50, 8, 300, 32])
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # query/key/value: torch.Size([50, 300, 256])

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout) # x:([50, 8, 300, 32]), attn:([50batch_size, 8heads, 300, 300])

        x = x.transpose(1, 2).contiguous() .view(nbatches, -1, self.h * self.d_k) # x: ([50, 300, 256])

        return self.linears[-1](x), torch.mean(self.attn,dim=0).cpu().detach().numpy() # x: ([50, 300, 256])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, input_d):
        super(Embeddings, self).__init__()
        self.proj = nn.Linear(input_d, d_model)

    def forward(self, x):
        return self.proj(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1) # [max_len,1] 
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

def make_model(src_d,tgt_d,N=6, d_model=512, d_ff=2048, h=8, dropout=0.1,norm_mel=False,encoder_only=False,spatial_attention=True):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_d),c(position)),
        nn.Sequential(Embeddings(d_model, tgt_d),c(position)),
        Generator(d_model,tgt_d,norm_mel),
        encoder_only=encoder_only
    )
    if spatial_attention:
        spatial_attention_layer=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 1) # 1 layer spatial attention
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_d), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_d), c(position)),
            Generator(d_model, tgt_d, norm_mel),
            encoder_only=encoder_only,
            spatial_attention_layer=spatial_attention_layer
        )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement lrate above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y,norm): 
        loss = self.criterion(x.contiguous(), y.contiguous())
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()