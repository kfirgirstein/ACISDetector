import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import json
ARCHITECTURES = ['alpha', 'amd64', 'arm64', 'armel', 'armhf', 'hppa', 'i386', 'ia64', 'm68k','mips', 'mips64el','mipsel', 'powerpc', 'powerpcspe', 'ppc64', 'ppc64el', 'riscv64', 's390', 's390x','sh4', 'sparc', 'sparc64', 'x32']

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

class RawBinaryDataset(Dataset):
    """Binary features dataset."""
    def __init__(self, json_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        with open(json_file,"r") as f:
            data = f.read()
            self.data_frame = json.loads(data)
            
        data_frame_X,data_frame_Y = self.__from_json_to_tensor__(self.data_frame)
        self.dataset =  TensorDataset(data_frame_X,data_frame_Y)
        
    def __from_json_to_tensor__(self,json_data:dict):
        x = []
        y = []
        for key,value in json_data.items():
            y.extend([ARCHITECTURES.index(key)]*len(value))
            x.extend(value)
            
        return torch.Tensor(x),torch.Tensor(y)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

def DatasetToTuple(sample):
    """Convert Dataset in sample to Tensors."""
       
    X_elem = []
    Y_elem = []
    for x,y in sample:
        X_elem.append(x if x.dim() > 0 else x.item())
        Y_elem.append(y if y.dim() > 0 else y.item())    
    return (torch.stack(X_elem),torch.stack(Y_elem))
def DatasetToTuple(sample):
    """Convert Dataset in sample to Tensors."""
       
    X_elem = []
    Y_elem = []
    for x,y in sample:
        X_elem.append(x if x.dim() > 0 else x.item())
        Y_elem.append(y if y.dim() > 0 else y.item())    
    return (torch.stack(X_elem),torch.stack(Y_elem))


