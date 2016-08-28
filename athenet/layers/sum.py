"""Sum layer."""

import numpy as np

import theano.tensor as T

from athenet.layers import Layer


class SumLayer(Layer):
    """Concatenation layer."""
    def __init__(self, input_layer_names=None, name='sum'):
        """Create sum layer.

        :param input_layer_names: List of input layers' names.
        """
        super(SumLayer, self).__init__(input_layer_names, name)
        self.input_shapes = None
        self._input_layers = None

    @property
    def input_layer_names(self):
        return self.input_layer_name

    @property
    def input_layers(self):
        return self._input_layers

    @input_layers.setter
    def input_layers(self, input_layers):
        self._input_layers = input_layers
        self.input_shapes = [layer.output_shape for layer in input_layers]
        for input_shape in self.input_shapes:
            if input_shape != self.input_shapes[0]:
                raise ValueError('all input layer sizes must match')

        self.input = [layer.output for layer in input_layers]
        train_input = [layer.train_output for layer in input_layers]
        if all([ti is not None for ti in train_input]):
            self.train_input = train_input

    @property
    def output_shape(self):
        return self.input_shape

    def _get_output(self, layer_inputs):
        """Return layer's output.

        :param layer_inputs: List of inputs in the format
                             (batch size, number of channels,
                              image height, image width).
        :return: Layer output.
        """
        return T.sum(layer_inputs, axis=0)
