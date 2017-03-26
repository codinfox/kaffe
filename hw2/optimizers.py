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

    def __init__(self, network, learning_rate = 0.01,
                 iterations = -1):
        """__init__

        :param network: the network to optimize
        :param learning_rate: learning rate
        :param iterations: maximum number of iterations, -1 is unlimited
        """
        super().__init__(network)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def optimize(self):
        i = 1
        net = self.network
        lr = self.learning_rate
        losses = []

        while True:
            loss = net.forward()
            losses.append(loss)
            print('Iteration {}, Loss = {}'.format(i, loss))
            net.backward()

            # update the weights
            for j in range(len(net.layers)):
                for k in net.params[j]:
                    net.params[j][k] -= lr * net.grads[j][k]

            i += 1
            if i > self.iterations and self.iterations != -1:
                break

        return losses
