import unittest
import layers
import numpy as np

class TestBase(unittest.TestCase):

    def setUp(self):
        self.params = {}
        self.grads = {}

    def loss(self, bottom):
        """ L2 loss (0.5*x^Tx)

        :param bottom: output from the layer
        :return: tuple (loss, grad)
        """
        inp = np.reshape(bottom, (-1,1))
        return (0.5 * inp.T.dot(inp), bottom)

    def checkGradientWrtX(self, layer, bottom_shape, eps,
            ana_grad_x, bottom, thresh):
        for idx, _ in np.ndenumerate(bottom):
            bottom_copy = bottom.copy()
            bottom_copy[idx] += eps # a new input
            out_1 = layer.forward(bottom_copy)
            loss_1, _ = self.loss(out_1)

            bottom_copy = bottom.copy()
            bottom_copy[idx] -= eps # a new input
            out_2 = layer.forward(bottom_copy)
            loss_2, _ = self.loss(out_2)

            neu_grad_x = (loss_1 - loss_2) * 0.5 / eps
            abs_diff = np.abs(neu_grad_x - ana_grad_x[idx])
            max_grad_x = np.max((np.abs(neu_grad_x), np.abs(ana_grad_x[idx])))

            self.assertLess(abs_diff / max_grad_x, 1e-5)

    def checkGradientWrtParams(self, layer, bottom_shape, eps,
            param_k, bottom, thresh):
        ana_grad = self.grads[param_k].copy()

        for idx, _ in np.ndenumerate(self.params[param_k]):
            param_copy = self.params[param_k].copy()
            self.params[param_k][idx] += eps
            out_1 = layer.forward(bottom)
            loss_1, _ = self.loss(out_1)

            param_copy[idx] -= eps # a new input
            self.params[param_k] = param_copy
            out_2 = layer.forward(bottom)
            loss_2, _ = self.loss(out_2)

            neu_grad = (loss_1 - loss_2) * 0.5 / eps
            abs_diff = np.abs(neu_grad - ana_grad[idx])
            max_grad = np.max((np.abs(neu_grad),
                np.abs(ana_grad[idx])))

            self.assertLess(abs_diff / max_grad, thresh)

    def checkGradient(self, layer, bottom_shape,
            param_key = None, eps = 10e-7, thresh = 1e-5):

        layer.setup(bottom_shape, self.params, self.grads)

        # random initialization
        bottom = np.random.rand(*bottom_shape)
        for k in self.params.keys():
            self.params[k] = np.random.rand(*self.params[k].shape)

        out = layer.forward(bottom) # forword layer for output
        _, top = self.loss(out)     # get gradient of loss

        # analytical gradients
        ana_grad_x = layer.backward(top)

        if param_key is None:
            self.checkGradientWrtX(layer, bottom_shape, eps,
                    ana_grad_x, bottom, thresh)
        else:
            self.checkGradientWrtParams(layer, bottom_shape,
                    eps, param_key, bottom, thresh)


class TestFullyConnectedLayer(TestBase):

    def testGradientX(self):
        layer = layers.FullyConnectedLayer(10)
        bottom_shape = (10,4)
        self.checkGradient(layer, bottom_shape)

    def testGradientW(self):
        layer = layers.FullyConnectedLayer(10)
        bottom_shape = (10,4)
        self.checkGradient(layer, bottom_shape, 'W')

    def testGradientb(self):
        layer = layers.FullyConnectedLayer(10)
        bottom_shape = (10,4)
        self.checkGradient(layer, bottom_shape, 'b')

class TestSigmoidActivationLayer(TestBase):

    def testGradientX(self):
        layer = layers.SigmoidActivationLayer()
        bottom_shape = (2, 1)
        self.checkGradient(layer, bottom_shape)

if __name__ == '__main__':
    unittest.main()

