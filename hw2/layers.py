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
        # don't count batch_size dimension
        n_input = reduce((lambda x,y: x * y), bottom_shape[1:])

        # setup the parameters
        params['W'] = np.zeros((n_input, self.n_neurons))
        params['b'] = np.zeros((self.n_neurons,))
        self.params = params

        # setup the grads
        grads['W'] = np.zeros((n_input, self.n_neurons))
        grads['b'] = np.zeros((self.n_neurons,))
        self.grads = grads

        # the output of fully connected layer can be thought of as 1x1 image
        return (bottom_shape[0], self.n_neurons, 1, 1)

    def forward(self, bottom):
        W = self.params['W']
        b = self.params['b']

        self.input = bottom
        # compress (channel, height, width) to one single dimension
        reshaped_bottom = bottom.reshape(bottom.shape[0], -1)
        self.output = (reshaped_bottom.dot(W) + b)
        # reshape to (batch_size, channel, height, width)
        self.output = self.output.reshape(
                reshaped_bottom.shape[0], self.n_neurons, 1, 1)

        return self.output

    def backward(self, top):
        W = self.params['W']
        b = self.params['b']

        reshaped_input = self.input.reshape(self.input.shape[0], -1)
        reshaped_top = top.reshape(top.shape[0], -1)

        dx = reshaped_top.dot(W.T).reshape(*self.input.shape)
        dW = reshaped_input.T.dot(reshaped_top)
        db = np.sum(reshaped_top, axis = 0).reshape(-1)

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
        self.output = None

    def setup(self, bottom_shape, params, grads):
        return bottom_shape

    def forward(self, bottom):
        # combine (channel, height, width) to one dimension, as our Softmax
        # only supports one single dimension
        reshaped_bottom = bottom.reshape(bottom.shape[0], -1)
        # we substract the max to avoid numerical issue
        normalized_bottom = reshaped_bottom - \
            np.max(reshaped_bottom, axis = 1).reshape(-1,1)
        exp = np.exp(normalized_bottom)
        self.output = exp / np.sum(exp, axis = 1).reshape(-1, 1)

        return self.output

    def backward(self, top):
        pass


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
        return (bottom_shape[0], 1, 1, 1)

    def forward(self, args):
        bottom, label = args
        self.label = label
        self.prob = self.softmax.forward(bottom)

        return -1.0 * np.sum(np.log(self.prob) * label) / bottom.shape[0]

    def backward(self, top = 1.0):
        return top * (self.prob - self.label)

class MNISTDataLayer(DataLayerBase):
    """ Layer to read and feed MNIST data to network
    """

    def __init__(self, path, batch_size = 20, shuffle = True):
        """ Initialize the layer

        :param path: path to the dataset
        """
        content = np.loadtxt(path, delimiter=',')
        N = content.shape[0]

        self.number_of_entries = N
        self.shuffle = shuffle
        self.batch_size = batch_size

        # reshape to (batch_size, channel, height, width)
        self.data = content[:,:-1].reshape(content.shape[0], 1, 28, 28)
        # translate digital label to one-hot label
        digit_label = content[:, -1].astype(np.int).reshape(-1)
        self.label = np.zeros((digit_label.shape[0], np.max(digit_label)+1))
        self.label[np.arange(digit_label.shape[0]), digit_label] = 1

        if shuffle:
            # generate shuffle index
            shuffle_idx = np.random.permutation(np.arange(N))
            # shuffle data
            self.data = self.data[shuffle_idx, :, :, :]
            # shuffle label
            self.label = self.label[shuffle_idx, :]

        self.idx = 0

    def setup(self, bottom_shape = None, params = None, grads = None):
        return (self.batch_size, 1, 28, 28)

    def forward(self, bottom = None):
        lo = self.idx
        hi = np.min((self.idx + self.batch_size, self.number_of_entries))
        out = self.data[lo:hi, :, :, :]
        lab = self.label[lo:hi, :]
        self.idx = hi
        if self.idx == self.number_of_entries:
            if self.shuffle:
                # re-shuffle
                shuffle_idx = np.random.permutation(
                        np.arange(self.number_of_entries))
                self.data = self.data[shuffle_idx, :, :, :]
                self.label = self.label[shuffle_idx, :]
            self.idx = 0
        return (out, lab)

    def backward(self, top):
        return None


class ReLUActivationLayer(LayerBase):
    """ ReLU activation function as a layer """

    def __init__(self):
        self.input = None

    def setup(self, bottom_shape, params, grads):
        return bottom_shape

    def forward(self, bottom):
        self.input = bottom

        return bottom * (bottom > 0.0).astype(np.float)

    def backward(self, top):
        return top * (self.input > 0.0).astype(np.float)


class DropoutLayer(LayerBase):
    """ DropoutLayer """

    def __init__(self, rate = 0.5, test = False):
        self.mask = None
        self.rate = 0.5
        self.test = test

    def setup(self, bottom_shape, params, grads):
        return bottom_shape

    def forward(self, bottom):
        out = None
        if not self.test:
            self.mask = (np.random.rand(*bottom.shape) > self.rate).astype(np.float)
            out = bottom * self.mask
        else:
            out = bottom * (1.0 - self.rate)

        return out

    def backward(self, top):
        return self.mask * top

