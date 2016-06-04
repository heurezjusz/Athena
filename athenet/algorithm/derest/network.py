import time
import os

from athenet.algorithm.derest.layers import get_derest_layer

class DerestNetwork(object):

    def __init__(self, network, *args):
        self.network = network

        d = os.path.dirname("tmp/")
        if not os.path.exists(d):
            os.makedirs(d)

        self.layers = [get_derest_layer(layer, i, *args)
                       for i, layer
                       in zip(xrange(len(network.layers)), network.layers)]

    def count_activations(self, inp):
        """
        Computes estimated activations for each layer

        :param Numlike inp: input of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike: possible output for network
        """
        for layer in self.layers:
            print time.time(), "count activation", type(layer)
            inp = layer.count_activation(inp)
            print time.time(), "Done"
        return inp

    def count_derivatives(self, outp):
        """
        Computes estimated impact of input of each layer on output of network

        :param Numlike outp: output of network
        :param boolean normalize: whenever normalize number between layers
        :return Numlike:
        """
        batches = outp.shape[0]
        for layer in reversed(self.layers):
            print time.time(), "count derivative", type(layer)
            outp = layer.count_derivatives(outp, batches)
            print time.time(), "Done"
        return outp

    def count_derest(self, count_function):
        """
        Returns indicators of each weight importance

        :param function count_function: function to count indicators,
            takes Numlike and returns float
        :return list of numpy arrays:
        """
        ""
        result = []
        for layer in self.layers:
            print time.time(), "count_derest", type(layer)
            indicators = layer.count_derest(count_function)
            result.extend(indicators)
            print time.time(), "Done", type(layer)
        return result
