# implementation of PINNsformer
# paper: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2307.11833

import torch
import torch.nn as nn
import pdb
from pinnsform.util import get_clones

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            nn.Tanh(),
            nn.Linear(d_ff, d_ff),
            nn.Tanh(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        
    def forward(self, x):
        x2 = self.act1(x)
        # pdb.set_trace()
        x = x + self.attn(x2,x2,x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = nn.Tanh()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = nn.Tanh()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


#class SpatioTemporalMixer(nn.Module):
#    def __init__(self, d_model):
#        super(SpatioTemporalMixer, self).__init__() 
#        self.linear = nn.Sequential(*[
#            nn.Linear(2, d_model),
#            WaveAct(),
#            nn.Linear(d_model, d_model),
#            WaveAct(),
#            nn.Linear(d_model, d_model)
#        ])
#
#    def forward(self, x):
#        return self.linear(x)


class PINNsformerTanh(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads):
        super(PINNsformerTanh, self).__init__()

        #self.st_mixer = SpatioTemporalMixer(d_model)
        self.st_mixer = nn.Linear(2, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out)
        ])

    def forward(self, src):
        #src = torch.cat((x,t), dim=-1)
        src = self.st_mixer(src)
        
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        
        output = self.linear_out(d_output)
        # pdb.set_trace()
        # raise Exception('stop')
        return output
