#!/usr/bin/env python3
#
# Simple class for two-layer fully connected neural networks.
#
# Date: February 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import math

import torch
import torch.nn as nn

SQRT2 = 1.414213562


def identity(x):
    """
    Identity function, can be used as an activation function after the second
    layer for committee machines.
    """
    return x


def erfscaled(x):
    """
    Re-scaled error function activation function.

    Useful to make the connection with an ODE description.
    """
    return torch.erf(x / SQRT2)


def getName(g):
    """
    Returns the name of the given activation function for the gain calculation
    for Xavier's initialisation.
    """
    name = None
    if g == torch.nn.functional.relu:
        name = "relu"
    elif g == erfscaled:
        name = "tanh"  # will probably do...
    elif g == identity:
        name = "linear"

    if name is None:
        raise ValueError("did not recognise g")

    return name


class TwoLayer(nn.Module):
    """
    Convenience class for fully connected two-layer neural network.
    """

    def __init__(
        self, g, N, num_units, num_cats, normalise1=False, normalise2=False, std0=None,
    ):
        """
        Constructs an instance of this two-layer network.

        Initial weights are drawn i.i.d. from a normal distribution with mean zero and
        variance chosen according to Xavier's method, unless it is explicitly given.

        Parameters:
        -----------

        g :
           if a single function, apply this non-linearity after each layer,
           including at the output layer. Can also be a tuple with the same
           number of activation functions as layers of weights (here, 2)
        N : input dimension
        num_units : number of hidden nodes
        num_cats : number of output nodes
        normalise1 :
           if True, pre-activations at the hidden layer are divided by sqrt(N)
        normalise2 :
           if True, pre-activations at the output layer divided by sqrt(K)
        std0 :
           Standard deviation of the initial (Gaussian) weights.

        """
        super(TwoLayer, self).__init__()
        self.N = N
        self.num_units = num_units
        self.num_cats = num_cats
        self.normalise1 = normalise1
        self.normalise2 = normalise2
        if isinstance(g, tuple):
            self.g1, self.g2 = g
        else:
            self.g1 = g
            self.g2 = g

        self.fc1 = nn.Linear(N, num_units, bias=False)
        self.fc2 = nn.Linear(num_units, num_cats, bias=False)

        if std0 is None:
            for i, w in enumerate([self.fc1.weight, self.fc2.weight]):
                name = getName(self.g1 if i == 0 else self.g2)
                gain = nn.init.calculate_gain(name)
                nn.init.xavier_normal_(w, gain=gain)
        else:
            nn.init.normal_(self.fc1.weight, mean=0.0, std=std0)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=std0)

    def forward(self, x):
        """
        Propagates the input x through the network.
        """
        # input to hidden
        norm = math.sqrt(self.N) if self.normalise1 else 1
        x = self.g1(self.fc1(x) / norm)

        # hidden to output
        norm = math.sqrt(self.num_units) if self.normalise2 else 1
        x = self.g2(self.fc2(x) / norm)
        return x

    def freeze(self):
        """
        Deactivates automatic differentiation for all parameters.

        Useful when defining a teacher.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Activates automatic differentiation for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

    def selfoverlap(self):
        """
        Returns the self-overlap matrix of the first layer of weights.
        """
        return self.fc1.weight.data.mm(self.fc1.weight.data.T) / self.N


class FlattenTransformation:
    """
    Flatten a torch.*Tensor representation of an image.

    Applications:
        Train stupid MLP models on image classification tasks.
    """

    def __init__(self, input_dimension):
        super(FlattenTransformation, self).__init__()
        self.N = input_dimension

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Transformed image.
        """
        return tensor.view(-1, self.N).squeeze()

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
