import numpy as np
import torch
import torch.nn

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

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out.to(self.device)