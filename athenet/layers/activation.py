"""Activation layers."""

import theano.tensor as T

from athenet.layers import Layer


class Activation(Layer):
    """Layer applying activation function to neurons."""
    def __init__(self, activation_function):
        """Create activation layer.

        :activation_function: Activation function to be applied.
        """
        super(Activation, self).__init__()
        self.activation_function = activation_function

    def _get_output(self, layer_input):
        return self.activation_function(layer_input)


def relu(x):
    """Rectified linear activation function.

    :x: Neuron input.
    """
    return T.maximum(0., x)


class ReLU(Activation):
    """Layer applying rectified linear activation function."""
    def __init__(self):
        super(ReLU, self).__init__(relu)
