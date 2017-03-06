from abc import ABC, abstractmethod

class OptimizerBase(ABC):
    """ The base class for all optimizers """

    def __init__(self, network):
        self.network = network

    @abstractmethod
    def optimize(self):
        """ Optimize the network """
        pass

class SGD(OptimizerBase):
    """ Stochastic Gradient Descent (SGD) optimizer """

    def __init__(self, network, learning_rate = 0.1):
        super().__init__(self, network)
        self.learning_rate = learning_rate

    def optimize(self):
        pass

