import os
import re
import sys
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import isadetect.helpers as isa_api 
import src.arch_v2_classifier as arch_api
import src.hyperparams as hp
from src.binary_dataset import FeatureDataset,DatasetToTuple,RawBinaryDataset,BinariesDataset
import src.arch_trainer as training


batch_size = 32
num_classes = 24

BINARIES_DATASET_DIR = "./dataset/binaries"
binaries_dataset = BinariesDataset(BINARIES_DATASET_DIR, 2500)

CHECKPOINTS_PATH = "./checkpoints/"
saved_state = torch.load(CHECKPOINTS_PATH + "rnn.pt", map_location=torch.device('cpu'))

rnn_hp = hp.rnn_hp()
model = arch_api.RNN(rnn_hp['i_s'],batch_size,num_classes,rnn_hp['l'],rnn_hp['h_f'])
model.load_state_dict(saved_state['model_state'])

N = len(binaries_dataset)
train_length = int(0.7* N)
test_length = N - train_length
ds_train,ds_test = torch.utils.data.random_split(binaries_dataset,(train_length,test_length))
dl_train = torch.utils.data.DataLoader(ds_train,batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test,batch_size=batch_size, shuffle=True)


ARCHITECTURES = ['alpha', 'amd64', 'arm64', 'armel', 'armhf', 'hppa', 'i386', 'ia64', 'm68k','mips', 'mips64el','mipsel', 'powerpc', 'powerpcspe', 'ppc64', 'ppc64el', 'riscv64', 's390', 's390x','sh4', 'sparc', 'sparc64', 'x32']

import json
for i in range(len(ARCHITECTURES)):
    json_data = {}
    for sample_idx in range(2500):
        if (sample_idx % 100 == 0):
            print(ARCHITECTURES[i] + ":" + str(sample_idx))
        stats = [0] * num_classes
        label = int(binaries_dataset[sample_idx + i*2500][1].item())
        if ARCHITECTURES[label] not in json_data:
            json_data[ARCHITECTURES[label]] = []
        data = binaries_dataset[sample_idx + i*2500][0]
        data = torch.split(data,1000)
        
        results = {}
        for block in range(len(data) - 1):
                result = model(data[block].reshape([1,1000])).tolist()
                results[str(block)] = result
        json_data[ARCHITECTURES[label]].append(results)

    with open("./json_datasets/" + ARCHITECTURES[i] + "_final_classification.json","w") as fj:
        json.dump(json_data,fj)
