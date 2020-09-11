import numpy as np
import torch
import itertools as it
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class CNN(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input e.g. (H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_h, in_w, = tuple(self.in_size)
        in_channels = 1
        layers = []
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        for i in range(len(self.channels)):
            layers.append(nn.Conv2d(in_channels, self.channels[i], 3, padding = 1))
            in_channels = self.channels[i]
            layers.append(nn.ReLU())
            if ((i+1)%self.pool_every == 0):
                layers.append(nn.MaxPool2d(2))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_h, in_w, = tuple(self.in_size)

        layers = []
        #  (Linear -> ReLU)*M -> Linear
        div_param = 2 ** (len(self.channels) // self.pool_every)
        layers.append(nn.Linear(int((in_h/div_param) * (in_w/div_param) * self.channels[-1]), self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

class ISADetectLogisticRegression(nn.Module):
    """
    A FC layer from features map to each of the classes followed by a softmax activation layer
    """

    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input
        :param out_classes: Number of classes to output in the final layer.
        """
        super().__init__()

        self.in_size = in_size
        self.out_classes = out_classes

        self.classifier = self._make_classifier()

    def _make_classifier(self):
        layers = []
        layers.append(nn.Linear(self.in_size, self.out_classes, bias = True))
        layers.append(nn.Softmax(dim = 1))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        out = self.classifier(x)
        return out
    

class MLP(nn.Module):
    """  
    Multi layer perceptron.
    All layers are fully conected.
    Relu after every layer and a softmax after the last layer.
    """

    def __init__(self, in_size, out_classes: int, hidden_dims: list):
        """
        :param in_size: Size of input e.g. (H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param hidden_dims: List of of length M containing hidden dimensions of the hidden layers
        """
        super().__init__()

        self.in_size = in_size
        self.out_classes = out_classes
        self.hidden_dims = hidden_dims

        self.classifier = self._make_classifier()


    def _make_classifier(self):
        self.in_size
        
        layers = []
        
        layers.append(nn.Linear(self.in_size, self.hidden_dims[0], bias = True))
        layers.append(nn.ReLU())
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1], bias = True))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes, bias = True))
        layers.append(nn.Softmax(dim = 1))            
            
        seq = nn.Sequential(*layers)
        
        return seq


    def forward(self, x):
        # x = x.flatten()
        out = self.classifier(x)
        return out
        
    
    
class RandomForest(nn.Module):
    """
    A Random Forest classifier model based on PyTorch nn.Modules.
    """

    def __init__(self, in_estimators, in_max_depth: list,
                 random_state: int, n_jobs: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()

        self.n_estimators = in_estimators
        self.max_depth = in_max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state , n_jobs= self.n_jobs)
        
    def __repr__(self):
        return f'RandomForest(estimators={self.n_estimators},max_depth={self.max_depth},random_state={self.random_state})'

    def forward(self,x):
        out = self.classifier.predict(x)
        return torch.FloatTensor(out)
    
    def fit(self, X,y):
        return self.classifier.fit(X,y)
    
    def evaluate(self,dl:  torch.utils.data.DataLoader,loss_fn,print_evey=100,):
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        
        dl_iter = iter(dl)
        for batch_idx in range(num_batches):
                data, y = next(dl_iter)
                y_hat = self.forward(data)
                current_loss = loss_fn(y_hat,y.float()).item()
                losses.append(current_loss)
                num_correct += (y_hat==y).sum().item()
                if batch_idx % print_evey ==  0:
                    print(batch_idx,"/",num_batches,": Loss ",current_loss,", Num Correct ",num_correct)
                
        avg_loss = sum(losses) / num_batches
        accuracy = 100. * num_correct / num_samples
        return {"losses": losses, "accuracy": accuracy}

    def test(self,sample,loss_fn,print_evey=10,):
        X,y = sample
        y_hat = torch.FloatTensor(self.classifier.predict(X))
        current_loss = loss_fn(y_hat,y.float())
        num_correct = (y_hat==y).sum()
        return {"losses": current_loss, "accuracy":  100. * num_correct / len(sample)}