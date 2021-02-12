# Núria Casals 950801-T740
# Robin de Groot 981116-T091


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    """ Create a feedforward neural network 

        Initialising weights: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    """
    def __init__(self, input_size, output_size, device):
        super(MyNetwork, self).__init__()

        self.device = device

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, 8)
        self.input_layer_activation = nn.ReLU()

        self.hidden1 = nn.Linear(8, 64)
        #self.hidden2 = nn.Linear(64,64)

        # Create output layer
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        x = self.input_layer(x)
        x = self.input_layer_activation(x)

        x = F.relu(self.hidden1(x))
        #x = F.relu(self.hidden2(x))
        # Compute output layer
        out = self.output_layer(x)
        return out.to(self.device)