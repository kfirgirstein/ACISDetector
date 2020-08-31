import numpy as np
import torch
import itertools as it
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
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
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
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
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
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

class ISADetectMLP(nn.Module):
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
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
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
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        return seq

    def forward(self, x):
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
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
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state , n_jobs= self.n_jobs , verbose=True)
        
    def forward(self, x,y):
        out = self.classifier.predict(x,y)
        return out
    
    def fit(self, x,y):
        self.classifier.fit(x,y)
