# baseline implementation of First Layer Wavelet

import torch
import torch.nn as nn


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)


class FLW(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(FLW, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(WaveAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        #src = torch.cat((x,t), dim=-1)
        return self.linear(x)


class FullWavelet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, seed = 0):
        super(FullWavelet, self).__init__()

        #generator = torch.Generator()
        #generator.manual_seed(int(seed+1))
        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(WaveAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(WaveAct())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        #src = torch.cat((x,t), dim=-1)
        return self.linear(x)


class FLLWaveletRest(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, seed = 0):
        super(FLLWaveletRest, self).__init__()

        #generator = torch.Generator()
        #generator.manual_seed(int(seed+1))
        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(WaveAct())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        #src = torch.cat((x,t), dim=-1)
        return self.linear(x)