"""Dropout layer."""

import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams

from athenet.layers import Layer


class Dropout(Layer):
    """Dropout layer."""
    def __init__(self, p_dropout=0.5):
        """Create dropout layer.

        :p_dropout: Weight dropout probability
        """
        super(Dropout, self).__init__()
        self.p_dropout = p_dropout

    def _get_output(self, layer_input):
        """Return layer's output.

        When evaluating, each weight is multiplied by (1 - p_dropout).

        :layer_input: Layer input.
        :return: Layer output.
        """
        return (1. - self.p_dropout) * layer_input

    def _get_train_output(self, layer_input):
        """Return layer's output used for training.

        When training, p_dropout weights are dropped out.

        :layer_input: Layer input.
        :return: layer output.
        """
        random = shared_randomstreams.RandomStreams()
        mask = random.binomial(n=1, p=1.-self.p_dropout,
                               size=layer_input.shape)
        return layer_input * T.cast(mask, theano.config.floatX)
