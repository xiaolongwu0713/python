import torch 
import torch.nn as nn 
import torch.nn.functional as F
import pywt
from torch import Tensor 
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

class Generator(nn.Module):
    def __init__(self, seq_len=150, channels=3, num_classes=9, latent_dim=100, data_embed_dim=10, 
                label_embed_dim=10 ,depth=3, num_heads=5, 
                forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.l1 = nn.Linear(self.latent_dim + self.label_embed_dim, self.seq_len * self.data_embed_dim)
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim) 
        
        self.blocks = Gen_TransformerEncoder(
                 depth=self.depth,
                 emb_size = self.data_embed_dim,
                 num_heads = self.num_heads,
                 drop_p = self.attn_drop_rate,
                 forward_drop_p=self.forward_drop_rate
                )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.data_embed_dim, self.channels, 1, 1, 0)
        )

        avgpool = nn.AvgPool2d((1, 9), stride=1, padding=(0, 4), count_include_pad=False)
        filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 41), padding=(0, 20),
                           padding_mode='reflect', groups=self.channels,bias=False) # depthwise


        w = pywt.Wavelet('bior6.8')
        (dec_lo, dec_hi, rec_lo, rec_hi) = w.filter_bank
        # interpolation
        tmp=[]
        for i in range(1,len(dec_lo)):
            tmp.append(dec_lo[i-1])
            m=(dec_lo[i-1]+dec_lo[i])/2
            tmp.append(m)
        #tmp.pop()
        # smoothing
        tmp2=[]
        for i in range(len(tmp)-1):
            tmp2.append((tmp[i]+tmp[i+1])/2)
        tmp2.append(tmp[-1])
        # expand from 34 to 41
        f=[]
        for i in range(4): f.append(tmp2[0])
        for i in range(len(tmp2)): f.append(tmp2[i])
        for i in range(3): f.append(tmp2[-1])

        f=np.asarray(f)
        f=f[np.newaxis,np.newaxis,np.newaxis,:]
        f=np.repeat(f,10,axis=0)


        filter.weight.data=torch.nn.Parameter(torch.Tensor(f)) # torch.Size([10, 1, 1, 19])
        self.last =filter # filter / nn.Sequential(avgpool,filter)

    def forward(self, z, labels):

        # aSection: share weight
        for i in range(self.last.weight.shape[0]):
            self.last.weight.data[i,:,:,:]=self.last.weight.data[0,:,:,:]

        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        x = self.l1(x)
        x = x.view(-1, self.seq_len, self.data_embed_dim)
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2)) # torch.Size([32, 10, 1, 500])

        #output=output.unsqueeze(2)
        #tmp=[]
        #for c in range(output.shape[1]):
        #    output[:,c,:,:]=self.last2(output[:,c,:,:].unsqueeze(1)).squeeze(1)
        output = self.last(output)


        return output # torch.Size([32, 10, 1, 500])


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)]) 
        

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

     

class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, adv_classes=2, cls_classes=10):
        super().__init__()
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, adv_classes)
        )
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes)
        )

    def forward(self, x):
        out_adv = self.adv_head(x)
        out_cls = self.cls_head(x)
        return out_adv, out_cls

    
class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))


    def forward(self, x:Tensor) ->Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x    
'''

'''        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=3,
                 patch_size=15,
                 data_emb_size=50,
                 label_emb_size=10,
                 seq_length = 150,
                 depth=3, 
                 n_classes=9, 
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, data_emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=data_emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(data_emb_size, 1, n_classes)
        )
        