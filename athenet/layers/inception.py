"""Inception Layer."""

import numpy as np

from athenet.layers import Layer, ConvolutionalLayer, ReLU, MaxPool, \
    Concatenation


class InceptionLayer(Layer):
    def __init__(self, n_filters, input_layer_name=None, name='inception'):
        """Create inception layer.

        Input for the Inception layer is in the format
        (batch size, number of channels, image height, image width).

        :param int list of length 6 n_filters: Number of filters in
                                               convolutional layers.
        """
        super(InceptionLayer, self).__init__(input_layer_name, name)

        layer_list1 = [
            ConvolutionalLayer(filter_shape=(1, 1, n_filters[0]),
                               name=name+'/1x1_conv1'),
            ReLU(),
        ]
        layer_list2 = [
            ConvolutionalLayer(filter_shape=(1, 1, n_filters[1]),
                               name=name+'/1x1_conv2'),
            ReLU(),
            ConvolutionalLayer(filter_shape=(3, 3, n_filters[2]),
                               padding=(1, 1),
                               name=name+'/3x3_conv'),
            ReLU(),
        ]
        layer_list3 = [
            ConvolutionalLayer(filter_shape=(1, 1, n_filters[3]),
                               name=name+'/1x1_conv3'),
            ReLU(),
            ConvolutionalLayer(filter_shape=(5, 5, n_filters[4]),
                               padding=(2, 2),
                               name=name+'/5x5_conv'),
            ReLU(),
        ]
        layer_list4 = [
            MaxPool(poolsize=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1)),
            ConvolutionalLayer(filter_shape=(1, 1, n_filters[5]),
                               name=name+'/1x1_conv4'),
            ReLU(),
        ]
        self.layer_lists = [layer_list1, layer_list2, layer_list3, layer_list4]
        layers = np.concatenate(self.layer_lists)
        self.convolutional_layers = [layer for layer in layers
                                     if isinstance(layer, ConvolutionalLayer)]
        self.bottom_layers = [layer_list[0] for layer_list in self.layer_lists]
        self.top_layers = [layer_list[-1] for layer_list in self.layer_lists]
        self.concat = Concatenation()

    @property
    def output_shape(self):
        return self.concat.output_shape

    @property
    def input_layer(self):
        return self._input_layer

    @input_layer.setter
    def input_layer(self, input_layer):
        self._input_layer = input_layer
        self.input_shape = input_layer.output_shape

        for bottom_layer in self.bottom_layers:
            bottom_layer.input_layer = input_layer

        for layers in self.layer_lists:
            for layer, prev_layer in zip(layers[1:], layers[:-1]):
                layer.input_layer = prev_layer
        self.concat.input_layers = self.top_layers

        self.output = self.concat.output
        self.train_output = self.concat.train_output

    def set_params(self, params):
        for layer, p in zip(self.convolutional_layers, params):
            layer.set_params(p)

    def get_params(self, params):
        return [layer.params for layer in self.convolutional_layers]
