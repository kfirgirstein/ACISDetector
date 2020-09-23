import os
import re
import sys
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

import isadetect.helpers as isa_api 
import src.arch_v2_classifier as arch_api
import src.hyperparams as hp
from src.binary_dataset import FeatureDataset,DatasetToTuple,RawBinaryDataset
import src.arch_trainer as training

import pathlib
import tarfile
import urllib
import shutil
DOWNLOAD_URL = 'https://github.com/kfirgirstein/ACISDetector/releases/download/Dataset/binary_raw.json.tar.gz'
DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')

def download_dataset(out_path=DATA_DIR, url=DOWNLOAD_URL, force=False):
    pathlib.Path(out_path).mkdir(exist_ok=True)
    out_filename = os.path.join(out_path, "ISAdetect_only_code_sections")
    tar_out_filename= out_filename + ".tar.gz"
    
    if os.path.isfile(tar_out_filename) and not force:
        print(f'Dataset file {tar_out_filename} exists, skipping download.')
    else:
        print(f'Downloading {url}...')
        with urllib.request.urlopen(url) as response, open(tar_out_filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f'Saved to {tar_out_filename}.')
    
    if os.path.isdir(out_filename) and not force:
        print(f'Dataset dir {out_filename} exists, skipping extraction.')
    else:
        tf = tarfile.open(tar_out_filename)
        tf.extractall()
        
    return out_filename


DATASET_FILE = "./dataset/binary_raw.json"
if not os.path.isfile(DATASET_FILE):
    DATASET_FILE = download_dataset()
binary_dataset = RawBinaryDataset(DATASET_FILE)

N = len(binary_dataset)
batch_size = 32
print(f'features length: {N}')

train_length = int(0.7* N)
test_length = N - train_length
ds_train,ds_test = torch.utils.data.random_split(binary_dataset,(train_length,test_length))
dl_train = torch.utils.data.DataLoader(ds_train,batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test,batch_size=batch_size, shuffle=True)

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

x0,y0 = ds_train[0]
dataset_shape = (x0.shape if x0.dim() > 0 else 1),(y0.shape if y0.dim() > 0 else 1)
print(x0.size(),y0.size())
print('input size =', dataset_shape[0], "X",dataset_shape[1] )

mlp_hp = hp.mlp_hp_raw()
print(mlp_hp)

_mlp = arch_api.MLP(in_size,num_classes,mlp_hp['hidden_size'],mlp_hp['dropout'])
optimizer = torch.optim.SGD(_mlp.parameters(), lr=mlp_hp["lr"],momentum=mlp_hp['momentum'])
loss_fn = torch.nn.CrossEntropyLoss()
trainer = training.ArchTrainer(_mlp, loss_fn, optimizer, device)
print(_mlp)

fit_res.append({"legend":"MLP","result":trainer.fit(dl_train,dl_test,num_epochs = 10,print_every=1)})
