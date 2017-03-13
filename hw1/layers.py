from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

class LayerBase(ABC):
    """ Abstract base class for layers """

    @abstractmethod
    def setup(self, bottom_shape, params, grads):
        """ Setup the layer and do some preparation

        This method will instantiate the parameters according to
        its own specification and bottom_shape.

        :param bottom_shape: the output shape from the previous layer
        :param params: the parameter entry in network
        :param grads: the gradient entry in network
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

        return (self.n_neurons, 1)

    def forward(self, bottom):
        W = self.params['W']
        b = self.params['b']

        self.input = bottom
        self.output = W.T.dot(bottom.reshape(-1,1)) + b

        return self.output

    def backward(self, top):
        W = self.params['W']
        b = self.params['b']

        dx = W.dot(top).reshape(*self.input.shape)
        dW = self.input.reshape(-1,1).dot(top.T).reshape(*W.shape)
        db = top.copy()

        self.grads['W'] = dW
        self.grads['b'] = db

        return dx


class SigmoidActivationLayer(LayerBase):
    """ Sigmoid activation function as a layer """

    def __init__(self):
        self.input = None
        self.output = None

    def setup(self, bottom_shape, params, grads):
        return bottom_shape

    def forward(self, bottom):
        self.input = bottom
        self.output = 1. / (1. + np.exp(-bottom))

        return self.output

    def backward(self, top):
        dx = top * self.output * (1. - self.output)

        return dx


class SoftmaxLayer(LayerBase):
    """ Softmax layer

    In Softmax layer, we always assume the input is a one dimensional vector
    (this is not a hard constraint, and softmax with other shapes are very
    useful as well in applications)
    """

    def __init__(self):
        self.input = None
        self.output = None

    def setup(self, bottom_shape, params, grads):
        return bottom_shape

    def forward(self, bottom):
        self.input = bottom
        # we substract the max to avoid numerical issue
        normalized_bottom = bottom - np.max(bottom)
        exp = np.exp(normalized_bottom)
        self.output = exp / np.sum(exp)

        return self.output

    def backward(self, top):
        # we don't implement this method here, but it is a very good exercise
        # to implement it yourself to see the difference this makes with the
        # SoftmaxCrossEntropyLossLayer
        pass


class CrossEntropyLayer(LayerBase):
    """ Cross entropy layer

    We don't implement this layer but recommend that you try to implement it
    yourself. This is a very good exercise. When completed, combine this and
    the Softmax layer to train a network. See the difference it makes with the
    SoftmaxCrossEntropyLossLayer
    """
    pass
