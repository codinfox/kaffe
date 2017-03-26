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


class DataLayerBase(LayerBase):
    """ Base class for data input layers

    Data Layers will does not have backward methods, and the forward method
    will return a tuple (data, label)
    """

    def backward(self, top):
        pass


class LossLayerBase(LayerBase):
    """ Base class for training loss layers

    Forward method of loss layers will take a tuple (bottom_data, label)
    instead of just bottom
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
        pass
        # sig = self.output
        # temp = top * sig
        # sum_temp = np.sum(temp)
        # out = temp - sig * sum_temp

        # return out


class CrossEntropyLossLayer(LossLayerBase):
    """ Cross entropy layer """

    def __init__(self):
        self.input = None
        self.output = None
        self.label = None

    def setup(self, bottom_shape, params, grads):
        return (1, 1)

    def forward(self, args):
        """ Forward method

        :param args: (bottom_data, label), label is a one-hot vector
        """
        bottom, label = args
        self.input = bottom
        self.label = label
        self.output = -np.sum(np.log(bottom) * label)

        return self.output

    def backward(self, top = 1.0):
        return -1.0 * top * self.label / self.input


class SoftmaxWithCrossEntropyLossLayer(LossLayerBase):
    """ Softmax with Cross Entropy Loss

    This layer combines softmax with Cross Entroy to provide numerically more
    stable solution. This layer only support one dimensional input, and the
    output is a scaler. This is just a design choice. It can be more general
    than this.

    Dunne, Rob A., and Norm A. Campbell.
    'On the pairing of the softmax activation and cross-entropy penalty functions
    and the derivation of the softmax activation function.'
    Proc. 8th Aust. Conf. on the Neural Networks, Melbourne, 181. Vol. 185. 1997.
    """

    def __init__(self):
        self.prob = None
        self.label = None
        self.softmax = SoftmaxLayer()

    def setup(self, bottom_shape, params, grads):
        return (1,1)

    def forward(self, args):
        bottom, label = args
        self.label = label
        self.prob = self.softmax.forward(bottom)

        return -np.sum(np.log(self.prob) * label)

    def backward(self, top = 1.0):
        return top * (self.prob - self.label)

class MNISTDataLayer(DataLayerBase):
    """ Layer to read and feed MNIST data to network
    """

    def __init__(self, path):
        """ Initialize the layer

        :param path: path to the dataset
        """
        content = np.loadtxt(path, delimiter=',')
        N = content.shape[0]

        self.number_of_entries = N

        self.data = content[:,:-1].T
        # translate digital label to one-hot label
        digit_label = content[:, -1].astype(np.int).reshape(-1)
        self.label = np.zeros((np.max(digit_label)+1, digit_label.shape[0]))
        self.label[digit_label, np.arange(digit_label.shape[0])] = 1

        # generate shuffle index
        shuffle_idx = np.random.permutation(np.arange(N))
        # shuffle data
        self.data = self.data[:, shuffle_idx]
        # shuffle label
        self.label = self.label[:, shuffle_idx]

        self.idx = 0

    def setup(self, bottom_shape = None, params = None, grads = None):
        return (self.data.shape[0], 1)

    def forward(self, bottom = None):
        out = self.data[:,[self.idx]]
        lab = self.label[:,[self.idx]]
        self.idx += 1
        if self.idx == self.number_of_entries:
            # re-shuffle
            shuffle_idx = np.random.permutation(
                    np.arange(self.number_of_entries))
            self.data = self.data[:, shuffle_idx]
            self.label = self.label[:, shuffle_idx]
            self.idx = 0
        return (out, lab)

    def backward(self, top):
        return None
