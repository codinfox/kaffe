import layers

class Network(object):
    """ Network wrapper that controls training and testing """

    def __init__(self, layers):
        """ Initiate the network with a list of layers

        :param layers: a list of configured layers
        """
        self.layers = layers
        self.grads = []
        self.params = []

        # setup the layers
        bottom_shape = None
        for layer in layers:
            # check to make sure the added layer is actually a layer
            assert(isinstance(layer, LayerBase))

            # we add empty dict to be filled by layer setup method
            self.params.append({})
            self.grads.append({})
            # it works because of pass by ref
            bottom_shape = layer.setup(bottom_shape, self.params[-1], self.grads[-1])

    def forward(self):
        """ Run forward pass across the network """
        pass

    def backward(self):
        """ Run backward pass across the network """
