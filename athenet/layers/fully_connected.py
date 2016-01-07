"""Fully connected layer."""

import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer


class FullyConnectedLayer(WeightedLayer):
    """Fully connected layer."""
    def __init__(self, n_out, n_in=None):
        """Create fully connected layer.

        n_out: Number of output neurons.
        n_in: Number of input neurons.
        """
        super(FullyConnectedLayer, self).__init__()
        self._n_in = None
        self.W_shared = None

        self.n_out = n_out
        self.n_in = n_in

    @property
    def n_in(self):
        """Return number of input neurons."""
        return self._n_in

    @n_in.setter
    def n_in(self, value):
        """Set number of input neurons."""
        if not value or self._n_in == value:
            return

        self._n_in = value

        W_value = np.asarray(
            np.random.normal(
                loc=0.,
                scale=np.sqrt(1. / self.n_out),
                size=(self.n_in, self.n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W_shared = theano.shared(W_value, borrow=True)

        b_value = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b_shared = theano.shared(b_value, borrow=True)

        self.params = [self.W_shared, self.b_shared]

    @property
    def input_shape(self):
        return self.n_in

    @input_shape.setter
    def input_shape(self, value):
        self.n_in = np.prod(value)

    @property
    def output_shape(self):
        """Return output shape."""
        return self.n_out

    def _reshape_input(self, raw_layer_input):
        """Return input in the format that is suitable for this layer.

        raw_layer_input: Input in the format (n_in, n_out) or compatible.
        """
        return raw_layer_input.flatten(2)

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return T.dot(self.input, self.W_shared) + self.b_shared

    def get_output_shape(self, input_shape):
        """Return output shape.

        input_shape: Input shape.
        """
        return input_shape
