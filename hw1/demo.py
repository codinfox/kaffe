import matplotlib.pyplot as plt

from network import Network
import layers
import optimizers

net = Network([
    layers.MNISTDataLayer(
        '/Users/zhli/Projects/kaffe/hw1/MNIST/MNIST_train.txt'),
    layers.FullyConnectedLayer(100),
    layers.SigmoidActivationLayer(),
    layers.FullyConnectedLayer(10),
    layers.SoftmaxWithCrossEntropyLossLayer()
    ])

losses = optimizers.SGD(net, iterations=200000).optimize()

def running_mean(l, N):
    """ Helper function, moving average

    :param l: list
    :param N: kernel size
    """
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result

plt.plot(running_mean(losses, 1000))
plt.show()
