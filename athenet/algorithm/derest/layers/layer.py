class DerestLayer(object):

    def __init__(self, layer):
        self.layer = layer
        self.activations = None
        self.derivatives = None

    def count_activation(self, layer_input, normalize=False):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        raise NotImplementedError

    def count_derivatives(self, layer_output, input_shape, normalize=False):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape:
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        raise NotImplementedError

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        return []
