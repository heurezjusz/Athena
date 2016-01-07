"""Softmax layer."""

import theano.tensor as T
from theano.tensor.nnet import softmax

from athenet.layers import Layer


class Softmax(Layer):
    """Softmax layer."""
    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return softmax(layer_input)

    def set_cost(self, y):
        """
        Set layer's cost variables.

        y: Desired output
        """
        self.cost = T.mean(-T.log(self.train_output)[T.arange(y.shape[0]), y])
