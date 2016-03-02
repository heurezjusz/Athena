"""Softmax layer."""

import theano.tensor as T
from theano.tensor.nnet import softmax

from athenet.layers import Layer


class Softmax(Layer):
    """Softmax layer."""
    def _get_output(self, layer_input):
        return softmax(layer_input)

    def set_cost(self, answer):
        """Set layer's cost function.

        answer: Vector of desired answers for minibatch.
        """
        self.cost = T.mean(-T.log(self.train_output)
                           [T.arange(answer.shape[0]), answer])
