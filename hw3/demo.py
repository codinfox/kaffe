import matplotlib.pyplot as plt
import numpy as np

from network import Network
import layers
import optimizers

net = Network([
    layers.MNISTDataLayer(
        '/Users/zhli/Projects/kaffe/MNIST/MNIST_train.txt'),
    layers.FullyConnectedLayer(100),
    layers.ReLUActivationLayer(),
    layers.DropoutLayer(),
    layers.FullyConnectedLayer(100),
    layers.ReLUActivationLayer(),
    layers.DropoutLayer(),
    layers.FullyConnectedLayer(10),
    layers.SoftmaxWithCrossEntropyLossLayer()
    ])

losses = optimizers.SGD(net, iterations=20000).optimize()

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

# testing
test_set_path = '/Users/zhli/Projects/kaffe/MNIST/MNIST_test.txt'
test_net = Network([
    layers.MNISTDataLayer(test_set_path, shuffle = False, batch_size = 1),
    layers.FullyConnectedLayer(100),
    layers.ReLUActivationLayer(),
    layers.DropoutLayer(test = True),
    layers.FullyConnectedLayer(100),
    layers.ReLUActivationLayer(),
    layers.DropoutLayer(test = True),
    layers.FullyConnectedLayer(10),
    layers.SoftmaxLayer()
    ])

test_net.restore(net.params)

test_set = np.loadtxt(test_set_path, delimiter=',')
test_label = test_set[:, -1].astype(np.int).reshape(-1)

correct = 0
for i in range(3000):
    out = test_net.forward()
    pred = np.argmax(out)
    if pred == test_label[i]:
        correct += 1
print('Accuracy:', correct * 1.0 / 3000)
