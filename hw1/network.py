import numpy as np
import layers

class Network(object):
    """ Network wrapper that controls training and testing

    The first layer will always provide data and labels for the whole network,
    and the last layer will always be optimization objective. This is just a
    design choice, not general.
    """

    def __init__(self, layers_list):
        """ Initiate the network with a list of layers

        :param layers: a list of configured layers
        """
        self.layers = layers_list
        self.grads = []
        self.params = []

        # setup the layers
        bottom_shape = None
        for layer in layers_list:
            # check to make sure the added layer is actually a layer
            assert(isinstance(layer, layers.LayerBase))

            # we add empty dict to be filled by layer setup method
            self.params.append({})
            self.grads.append({})
            # it works because of pass by ref
            bottom_shape = layer.setup(bottom_shape, self.params[-1],
                                       self.grads[-1])

        # Initialize weight
        def instantiate_weights(input_size, output_size):
            """ Initialize the weights for network based on heuristics

            :param input_size: how many neurons in previous layer
            :param output_size: how many neurons in current layer
            """
            b = np.sqrt(6) / np.sqrt(input_size + output_size)
            return np.random.uniform(-b, b, (input_size, output_size))

        for j in range(len(layers_list)):
            for k in self.params[j]:
                if k == 'b':
                    # we don't initialize bias
                    continue
                self.params[j][k] += instantiate_weights(
                        self.params[j][k].shape[0], self.params[j][k].shape[1])

    def forward(self):
        """ Run forward pass across the network """
        bottom = None
        for layer in self.layers:
            # a design I'm not proud of
            if isinstance(layer, layers.DataLayerBase):
                bottom, label = layer.forward(bottom)
            elif isinstance(layer, layers.LossLayerBase):
                bottom = layer.forward((bottom, label))
            else:
                bottom = layer.forward(bottom)

        return bottom

    def backward(self):
        """ Run backward pass across the network """
        top = 1.0
        for layer in reversed(self.layers):
            top = layer.backward(top)
