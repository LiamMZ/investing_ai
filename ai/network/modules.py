import logging
import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as  F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, device="cpu"):
        super(Chomp1d, self).__init__()
        self.device = device
        self.chomp_size = chomp_size
        logging.info("Created Chomp1d Module")

    def forward(self, x):
        logging.debug('Chomp1d Input - {}'.format(x.shape))
        output = x[:, :, :-self.chomp_size].contiguous()
        logging.debug('Chomp1d Output - {}'.format(output.shape))
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, device="cpu"):
        super(TemporalBlock, self).__init__()
        self.device = device
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride,padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, device=self.device)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        self.chomp2 = Chomp1d(padding, device=self.device)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.elu1, self.dropout1,
                                 self.conv2, self.chomp2, self.elu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.elu = nn.ELU()
        logging.info('Created Temporal Block Module')

    def forward(self, x):
        logging.debug('Temporal Block Input - {}'.format(x.shape))
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        output = self.elu(out + res)
        logging.debug('Temporal Block Output - {}'.format(output.shape))

class AttentionBlock(nn.Module):
    def __init__(self, input_dims, k_size, vsize, device="cpu"):
        super(AttentionBlock, )