import pickle
import re
import os
import ray
import gc
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.config import normal_ranges, diseaselabs


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_and_normalization(data, lowcut, highcut, fs, order=5, normalization=False):
    yall = []
    for dat in data:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        bp = lfilter(b, a, dat)
        if normalization:
            # Normalize between -1 and 1vscode-remote://ssh-remote%2B7b22686f73744e616d65223a223134372e34362e3136322e313737222c2275736572223a226a756e6d6f6b696d5f393631323238222c22706f7274223a31303230357d/home/jisoolee_991011/Projects/YOON/dai/data_parsing.ipynb
            y = 2*(bp - np.min(bp)) / (np.max(bp)-np.min(bp))
            bp = y - 1
        yall.append(bp)
    
    return np.array(yall)


def seedset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class ECG_Dataset(Dataset):
    def __init__(self, conti_data, disease):
        self.conti_data = conti_data.set_index('PT_NO')
        self.conti_data['age'] = self.conti_data['age'] / 40 - 1
        self.conti_data['gender'] = self.conti_data['gender'].apply(lambda x: 1 if x == 'MALE' else 0)
        
        self.lead8 = torch.cat([torch.Tensor(i) for i in tqdm(self.conti_data['waveform'])]).view(-1, 12, 5000)
        self.agesex = torch.Tensor(self.conti_data[['age', 'gender']].values)
        self.flag = torch.Tensor(self.conti_data[[f'{disease}_flag']].values)
            
    def __len__(self):
        return self.conti_data.shape[0]
    
    def __getitem__(self, idx):
        # Return Age, Sex, Signal
        return self.lead8[idx], self.agesex[idx], self.flag[idx]



def dataload(conti, args):
    conti.reset_index(inplace=True, drop=True)

    trainidx, testidx = train_test_split(
        pd.unique(conti['PT_NO']), test_size=0.4, random_state=args.seed
    )
    validx, testidx = train_test_split(
        testidx, test_size=0.5, random_state=args.seed
    )

    ds = ECG_Dataset(conti[conti['PT_NO'].isin(trainidx)], args.disease)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True)
    del ds
    gc.collect()

    vds = ECG_Dataset(conti[conti['PT_NO'].isin(validx)], args.disease)
    vdl = DataLoader(vds, batch_size=args.bs, shuffle=False)
    del vds
    gc.collect()

    tds = ECG_Dataset(conti[conti['PT_NO'].isin(testidx)], args.disease)
    tdl = DataLoader(tds, batch_size=args.bs, shuffle=False)
    del tds
    gc.collect()

    return dl, vdl, tdl



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device


    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(input, target, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float).to(self.device)
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(target)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(target)] = alpha.index_select(0, torch.unique(target))
                alpha_t = temp.gather(0, target)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, target)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss