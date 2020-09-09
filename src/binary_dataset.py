import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset

class FeatureDataset(Dataset):
    """Binary features dataset."""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data_frame = np.genfromtxt(csv_file, delimiter=',', skip_header=True, filling_values=0)
        self.data_frame = torch.from_numpy(self.data_frame)
        data_frame_X = self.data_frame[:,0:-1]
        data_frame_Y = self.data_frame[:,-1]
        self.dataset =  TensorDataset(data_frame_X,data_frame_Y)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
