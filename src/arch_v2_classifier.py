import numpy as np
import torch
import itertools as it
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class CNN(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.
    This CNN is made out of the specified amount of convolutional layers endded by a fully connected layer

    The architecture is:
    [(CONV -> ReLU)* layers -> Linear
    """

    def __init__(self, in_size, out_classes: int, hidden_channels: list, kernel_size, stride, padding = 0, ):
        """
        :param in_size: Size of input e.g. (Length).
        :param out_classes: Number of classes to output in the final FC layer.
        :param hidden_channels: A list of channels, this will determine the amount of convolution layers
        :param kernel_size: P, the number of conv layers before each max-pool.
        :param stride:  CNN stride
        :param padding: CNN padding, logic says it should be 0
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

class RNN(nn.Module):
    """
    An RNN layer followed by a FC layer
    """

    def __init__(self, in_size, batch_size, out_classes: int, num_layers = 1, hidden_features = 256):
        """
        :param in_size: Size of input
        :param batch_size: Number of batches in the input.
        :param out_classes: Number of classes to output in the final layer.
        :param num_layers: Number of stacked RNN layers.
        :param hidden_features: Number of features in an RNN hidden layer state.
        """
        super().__init__()

        self.in_size = in_size
        self.out_classes = out_classes
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_features = hidden_features

        # RNN part
        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_features, num_layers = self.num_layers, nonlinearity = 'tanh', bias = True )
            
        # Labeling part 
        label_layers = []
        layers.append(nn.Linear(self.num_layers * hidden_size, self.out_classes, bias = True))
        layers.append(nn.Softmax(dim = 1)) 
        self.label = nn.Sequential(*layers)


    def forward(self, x):
        '''
        It is assumed x is of size (Batch_size x sequence_length )
        '''
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_features)
        _, h_n = self.RNN(x, h_0)
        out = self.label(h_n)
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
        
    