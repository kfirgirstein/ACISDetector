import numpy as np
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.
    This CNN is made out of the specified amount of convolutional layers endded by a fully connected layer

    The architecture is:
    [(CONV -> ReLU)* layers -> Linear
    """

    def __init__(self, in_size, out_classes: int, kernel_size,stride,dilation=1,hidden_channels = None, padding = 0):
        """
        :param in_size: Size of input e.g. (Length).
        :param out_classes: Number of classes to output in the final FC layer.
        :param hidden_channels: A list of channels, this will determine the amount of convolution layers
        :param kernel_size: P, the number of conv layers before each max-pool.
        :param stride:  CNN stride
        :param padding: CNN padding, logic says it should be 0
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        
        
    def _make_feature_extractor(self):
        '''
        (Conv1d -> RelU) * hidden_channels_times
        '''
        in_len = self.in_size
        cnn_layers = []
        if self.hidden_channels is None:
            cnn_layers.append(nn.Conv1d(1, 1, self.kernel_size, self.stride, self.padding, self.dilation, bias = True))
            cnn_layers.append(nn.ReLU())
            self.out_len = self._compute_len_after_convolution(in_len,  self.kernel_size,  self.dilation,  self.stride, self.padding)
        else:
            in_channels = 1
            for i in (range(len(self.hidden_channels) - 1)):
                cnn_layers.append(nn.Conv1d(in_channels, self.hidden_channels[i], self.kernel_size, self.stride, self.padding, self.dilation, bias = True))
                cnn_layers.append(nn.ReLU())
                in_channels = self.hidden_channels[i]

                out_len = self._compute_len_after_convolution(in_len,self.kernel_size,  self.dilation,  self.stride, self.padding)
                in_len = out_len
                
            cnn_layers.append(nn.Conv1d(in_channels, self.hidden_channels[-1], self.kernel_size, self.stride, self.padding, self.dilation, bias = True))
            cnn_layers.append(nn.ReLU())
            self.out_len = self._compute_len_after_convolution(in_len,self.kernel_size,  self.dilation,  self.stride, self.padding)
            
        cnn = nn.Sequential(*cnn_layers)
        return cnn
        

    def _make_classifier(self):
        '''
        (Linear -> Softmax) - one FC layer
        '''
        layers = []
        if self.hidden_channels is None:
            layers.append(nn.Linear(self.out_len, self.out_classes))
        else:
            layers.append(nn.Linear(self.out_len * self.hidden_channels[-1], self.out_classes))
        layers.append(nn.Softmax(dim = 1))
        seq = nn.Sequential(*layers)
        return seq

        
    def _compute_len_after_convolution(self,in_len, kernel_size, dilation, stride, padding):
        out_len = ((in_len + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1
        return out_len


    def forward(self, x):
        '''
        x is of size (batch_size, sequence_length)
        Extract features from the input, run the classifier on them and
        return class scores.
        '''
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
        self.rnn = nn.RNN(input_size = self.in_size, hidden_size = self.hidden_features, num_layers = self.num_layers, nonlinearity = 'tanh', bias = True )
            
        # Labeling part 
        label_layers = []
        label_layers.append(nn.Linear(self.num_layers *  self.hidden_features, self.out_classes, bias = True))
        label_layers.append(nn.Softmax(dim = 1)) 
        self.label = nn.Sequential(*label_layers)


    def forward(self, x):
        '''
        It is assumed x is of size (Batch_size, sequence_length )
        '''
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_features)
        _, h_n = self.rnn(x, h_0)
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
        
    