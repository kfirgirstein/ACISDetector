import numpy as np
import torch
import json
from torch.utils.data import TensorDataset, DataLoader,Dataset
import os


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
            json_file (string): Path to the json file the 1 Kb samples.
        """
        with open(json_file,"r") as f:
            data = f.read()
            self.data_frame = json.loads(data)
            
        data_frame_X,data_frame_Y = self._from_json_to_tensor(self.data_frame)
        self.dataset =  TensorDataset(data_frame_X,data_frame_Y)
        
    def _from_json_to_tensor(self, json_data:dict):
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


class BinariesDataset(Dataset):
    def __init__(self, binaries_path, n_binaries_from_arch):
        """
        Args:
            binaries_path (string): Path to the dir with all the binaries.
            n_binaries_from_arch (int): The amount of binaries to take from each arch
        """
        self.n_binaries_from_arch = n_binaries_from_arch
        self.dataset = []
        
        # Make sure the dataset exists
        if (os.path.exists(binaries_path)):
            for arch in ARCHITECTURES:
                arch_binaries_path = binaries_path + "/" + arch
                if (os.path.exists(arch_binaries_path)):
                    
                    # First we get all files in dir
                    files = os.listdir(arch_binaries_path)
                    if len(files) < self.n_binaries_from_arch:
                        print("There are not enough binaries from " + arch + " architecture")
                        raise BinariesMissingError

                    # Next, we append all file paths into the dataset
                    files_appended = 0
                    for i in range(len(files)):
                        curr_file_path = arch_binaries_path + "/" + files[i]
                        if os.stat(curr_file_path).st_size >= 1000:
                            self.dataset.append((arch, curr_file_path))
                            files_appended += 1
                        if(files_appended == self.n_binaries_from_arch):
                            break

                    # Finally, we make sure enough files were appended
                    if (files_appended < self.n_binaries_from_arch):
                        print(arch + " binaries folder is missing")
                        raise BinariesFolderMissingError                       
                else:
                    print(arch + " binaries folder is missing")
                    raise BinariesFolderMissingError                
        else:
            print("Binaries folder is missing")
            raise BinariesFolderMissingError
    
    
    def __len__(self):
        return len(ARCHITECTURES) * self.n_binaries_from_arch

    def __getitem__(self, idx):
        file_path = self.dataset[idx][1]
        arch = self.dataset[idx][0]
        return (torch.Tensor(np.fromfile(file_path, dtype='uint8')), torch.Tensor([ARCHITECTURES.index(arch)]))


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


class ClassifiedDataset(Dataset):
    def __init__(self, binaries_path, archs_classified, n_binaries_from_arch, n_classes):
        """
        Args:
            binaries_path (string): Path to the dir with all the binaries.
            n_binaries_from_arch (int): The amount of binaries to take from each arch
        """
        self.n_binaries_from_arch = n_binaries_from_arch
        self.binaries_path = binaries_path
        self.n_classes = n_classes
        self.dataset = {}
        self.archs_classified = archs_classified
        
        if (os.path.exists(binaries_path)):
            for arch in self.archs_classified:
                json_file = open(binaries_path + arch + "_final_classification.json", 'rb')
                data = json.load(json_file)
                self.dataset.update(data)
                json_file.close()
    
    
    def __len__(self):
        return len(self.dataset) * self.n_binaries_from_arch

    def __getitem__(self, idx):
        cnt = 0 
        arch = self.archs_classified[idx//self.n_binaries_from_arch]
        label = ARCHITECTURES.index(arch)
        classifications = self.dataset[arch][idx % self.n_binaries_from_arch]
        while(len(classifications) == 0):
            cnt += 1
            classifications = self.dataset[arch][(idx+cnt) % self.n_binaries_from_arch ]
        most_popular = [0] * self.n_classes
        percentage = [0] * self.n_classes
        
        for i in range(len(classifications)):
            result  = classifications[str(i)][0]
            most_popular[np.argmax(result)] += 1
            percentage = np.add(percentage, np.array(list(result), dtype=int))
            
        most_popular = np.divide(most_popular, float(len(classifications)))
        percentage = np.divide(percentage, float(len(classifications)))
        #y = [0] * self.n_classes
        #y[label] += 1
        #return (torch.Tensor(np.append(percentage, most_popular)), torch.Tensor([y]))
        return (torch.Tensor(np.append(percentage, most_popular)), np.argmax(result))