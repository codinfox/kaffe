from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

class LayerBase(ABC):
    """ Abstract base class for layers """

    @abstractmethod
    def setup(self, bottom_shape, params):
        """ Setup the layer and do some preparation

        This method will instantiate the parameters according to
        its own specification and bottom_shape.

        :param params: the parameter entry in network
        :return: the output size of the current layer
        """
        pass

    @abstractmethod
    def forward(self, bottom):
        """ Forward pass for the layer

        :param bottom: the output from the bottom layer
        :return: the output from the current layer to feed to next layer
        """
        pass

    @abstractmethod
    def backward(self, top):
        """ Backward pass for the layer

        :param top: the derivative (output of backward method) w.r.t. input
                    from the top layer
        :return: a tuple that contains gradients w.r.t. input and parameters
        """
        pass


class FullyConnectedLayer(LayerBase):
    """ Fully connected layer that implements full connections """

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.params = None
        self.grads = None
        self.input = None
        self.output = None

    def setup(self, bottom_shape, params, grads):
        n_input = reduce((lambda x,y: x * y), bottom_shape)

        # setup the parameters
        params['W'] = np.zeros((n_input, self.n_neurons))
        params['b'] = np.zeros((self.n_neurons, 1))
        self.params = params

        # setup the grads
        grads['W'] = np.zeros((n_input, self.n_neurons))
        grads['b'] = np.zeros((self.n_neurons, 1))
        self.grads = grads

        return (n_neurons, 1)

    def forward(self, bottom):
        W = self.params['W']
        b = self.params['b']

        self.input = bottom
        self.output = W.T.dot(bottom) + b

        return self.output

    def backward(self, top):
        W = self.params['W']
        b = self.params['b']

        dx = top.T.dot(W)
        dW = self.input.dot(top.T)
        db = top.copy()

        self.grads['W'] = dW
        self.grads['b'] = db

        return dx

class SigmoidActivationLayer(LayerBase):
    """ Sigmoid activation function as a layer """

    def __init__(self):
        self.input = None
        self.output = None

    def setup(self, bottom_shape, params):
        return bottom_shape

    def forward(self, bottom):
        self.input = bottom
        self.output = 1. / (1. + np.exp(-bottom))

        return self.output

    def backward(self, top):
        dx = top * self.input * (1. - self.input)

        return dx
