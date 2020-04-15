# -*- coding: utf-8 -*-
"""
Model for Variational Autoencoder

@author: Yash Gupta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,net_list):
        super(VAE, self).__init__()
        self.d = len(net_list)
        assert(self.d>=2)
        layer_info = {}
        layers = {}
        #encoder
        for i in range(self.d-2):
            lname = 'linear_enc'+str(i+1)
            layer_info[lname] = (net_list[i],net_list[i+1])
            layers[lname] = nn.Linear(net_list[i],net_list[i+1])

        #sampling
        lname = 'linear_enc'+str(self.d)+'_mean'
        layer_info[lname] = (net_list[self.d-2],net_list[self.d-1])
        layers[lname] =  nn.Linear(net_list[self.d-2],net_list[self.d-1])
        
        lname = 'linear_enc'+str(self.d)+'_std'
        layer_info[lname] = (net_list[self.d-2],net_list[self.d-1])
        layers[lname] = nn.Linear(net_list[self.d-2],net_list[self.d-1])
        
        #decoder
        for i in reversed(range(self.d-1)):
            lname = 'linear_dec'+str(i+1)
            layer_info[lname] = (net_list[i+1],net_list[i])
            layers[lname] = nn.Linear(net_list[i+1],net_list[i])
        
        self.layers_info = layer_info
        self.layers = nn.ModuleDict(layers)
        
    def describe(self):
        for k,v in self.layers_info.items():
            print(k,v)
            
    def encode(self,x):
        for i in range(self.d-2):
            x = F.relu(self.layers['linear_enc'+str(i+1)](x))
        return self.layers['linear_enc'+str(self.d)+'_mean'](x),self.layers['linear_enc'+str(self.d)+'_std'](x)
    
    def decode(self,z):
        for i in reversed(range(1,self.d-1)):
            z = F.relu(self.layers['linear_dec'+str(i+1)](z))
        return torch.sigmoid(self.layers['linear_dec'+str(1)](z))
    
    def reParTrick(self,mean,logstd):
        #log(std) is the ouptput
        std = torch.exp(logstd)
        eps = torch.randn_like(logstd)
        return mean + eps*std
    
    def forward(self, x):
        mean, logstd = self.encode(x)
        z = self.reParTrick(mean, logstd)
        xx = self.decode(z)
        return xx, mean, logstd
    
def VAEloss(xx, x, mean, logstd, beta=1):
    recon_loss = F.binary_cross_entropy(xx, x, reduction='sum')
    kld_loss = 0.5 * torch.sum(torch.exp(logstd) + mean**2 - 1. - logstd)
    return recon_loss + beta*kld_loss
