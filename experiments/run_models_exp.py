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
from src.binary_dataset import FeatureDataset, DatasetToTuple, RawBinaryDataset
import src.arch_trainer as training

RESULT_DIR = "./experiments"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
os.makedirs(RESULT_DIR, exist_ok=True)

import pathlib
import tarfile
import urllib
import shutil

DOWNLOAD_URL = "https://github.com/kfirgirstein/ACISDetector/releases/download/Dataset/binary_raw.json.tar.gz"
DATA_DIR = pathlib.Path.home().joinpath(".pytorch-datasets")


def download_dataset(out_path=DATA_DIR, url=DOWNLOAD_URL, force=False):
    pathlib.Path(out_path).mkdir(exist_ok=True)
    tar_out_filename = os.path.join(out_path, os.path.basename(url))
    out_filename = tar_out_filename[: tar_out_filename.find(".tar")]

    if os.path.isfile(out_filename) and not force:
        print(f"Dataset file {out_filename} exists, skipping download.")
    else:
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response, open(
            tar_out_filename, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Saved to {tar_out_filename}.")

        tf = tarfile.open(tar_out_filename)
        tf.extractall(out_path)
        print(f"All dataset extracte\nYou can start working!.")
    return out_filename


DATASET_FILE = "./dataset/binary_raw.json"
if not os.path.isfile(DATASET_FILE):
    DATASET_FILE = download_dataset()
binary_dataset = RawBinaryDataset(DATASET_FILE)

from jupyter_utils.plot import plot_fit, plot_exp_results
import experiments.run_ex as experiments

fig = None
fit_res = []
N = len(binary_dataset)
batch_size = 64
num_classes = 23

print(f"features length: {N}")
train_length = int(0.7 * N)
test_length = N - train_length
ds_train, ds_test = torch.utils.data.random_split(
    binary_dataset, (train_length, test_length)
)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True)
print(f"Train: {len(ds_train)} samples")
print(f"Test: {len(ds_test)} samples")
x0, y0 = ds_train[0]
dataset_shape = (x0.shape if x0.dim() > 0 else 1), (y0.shape if y0.dim() > 0 else 1)
in_size = dataset_shape[0][0]

## MLP

ex_hidden_dim = [[300, 700], [700, 300], [2000, 4000, 8000], [16000], [8000, 4000, 2000]]
ex_dropout = [0.0, 0.3, 0.5, 0.7]
print(f"****************************\n\nNow let's train MLP\nhyperparameter are:\nHidden Dimension: {ex_hidden_dim}\nDropout: {ex_dropout}")
for h_d in ex_hidden_dim:
    for d in ex_dropout:
         exp_name = f"exp_MLP_d{d}_hd{'_'.join(map(str,h_d))}"
            if os.path.isfile(f"{os.path.join(RESULT_DIR, exp_name)}.json"):
                continue
        _mlp = arch_api.MLP(
            in_size, num_classes, h_d, d
        )
        optimizer = torch.optim.SGD(
            _mlp.parameters(), lr=mlp_hp["lr"], momentum=mlp_hp["momentum"]
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = training.ArchTrainer(_mlp, loss_fn, optimizer, device)
        experiments.run_experiment(exp_name, trainer, dl_train, dl_test, num_epochs=30, print_every=50)
print(f"****************************\n\n Wow it took a while ... to see the results you can go into to {RESULT_DIR} folder and look for all MLP experiments!")
## CNN

ex_stride = [4, 6, 8]
ex_kernel = [4, 8, 16, 32]
ex_channels = [[1, 1, 4], [1, 2, 4, 8], [10, 100, 200]]
print(f"****************************\n\nNow let's train CNN\nhyperparameter are:\nStride: {ex_stride}\nKernel: {ex_kernel}\nHidden Channels: {ex_channels}")
for s in ex_stride:
    for k in ex_kernel:
        for h_c in ex_channels:
            exp_name = f"exp_CNN_S{s}_k{k}_L{'_'.join(map(str,h_c))}"
            if os.path.isfile(f"{os.path.join(RESULT_DIR, exp_name)}.json"):
                continue
            _cnn = arch_api.CNN(
                in_size,
                num_classes,
                kernel_size=k,
                stride=s,
                dilation=cnn_hp["d"],
                hidden_channels=h_c,
                padding=cnn_hp["p"],
            )
            optimizer = torch.optim.Adam(_cnn.parameters(), lr=cnn_hp["lr"])
            loss_fn = torch.nn.CrossEntropyLoss()
            trainer = training.ArchTrainer(_cnn, loss_fn, optimizer, device)
            experiments.run_experiment(
                exp_name, trainer, dl_train, dl_test, num_epochs=30, print_every=50
            )
print(f"****************************\n\n Wow it took a while ... to see the results you can go into to {RESULT_DIR} folder and look for all CNN experiments!")

## RNN
ex_input_size = [25, 50, 100, 250, 500, 1000]
ex_hidden_features = [16, 32, 64, 128, 256, 512]
ex_num_layers = [2, 4, 8, 16]
print(f"****************************\n\nNow let's train RNN\nhyperparameter are:\nInput Size: {ex_hidden_features}\nNumber of layers: {ex_num_layers}\nHidden Features: {ex_hidden_features}")
for i_s in ex_input_size:
    for h_f in ex_hidden_features:
        for l in ex_num_layers:
            exp_name = f"exp_RNN_is{i_s}_hf{h_f}_l{l}"
            if os.path.isfile(f"{os.path.join(RESULT_DIR, exp_name)}.json"):
                continue
            _rnn = arch_api.RNN(i_s, batch_size, num_classes, l, h_f)
            optimizer = torch.optim.Adam(_rnn.parameters(), lr=rnn_hp["lr"])
            loss_fn = torch.nn.CrossEntropyLoss()
            trainer = training.ArchTrainer(_rnn, loss_fn, optimizer, device)
            experiments.run_experiment(
                exp_name, trainer, dl_train, dl_test, num_epochs=30, print_every=50
            )

print(f"****************************\n\n Wow it took a while ... to see the results you can go into to {RESULT_DIR} folder and look for all CNN experiments!")

print("****************************\n\n Thanks you and Bye Bye!\n@ACISDetectorTeam")