"""Convolutional layer."""

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from athenet.layers import WeightedLayer


class ConvolutionalLayer(WeightedLayer):
    """Convolutional layer."""
    def __init__(self, filter_shape, image_shape=None, stride=(1, 1),
                 padding=(0, 0), n_groups=1, batch_size=1):
        """Create convolutional layer.

        filter_shape: Filter shape in the format
                      (filter height, filter width, number of filters).
        image_shape: Image shape in the format
                     (image height, image width, number of channels).
        stride: Tuple representing interval at which to apply the filters.
        padding: Tuple representing number of zero-valued pixels to add on
                 each side of the input.
        n_groups: Number of groups which input and output channels will be
                  split into. Two channels are connected only if they belong
                  to the same group.
        batch_size: Minibatch size.
        """
        super(ConvolutionalLayer, self).__init__()
        self._image_shape = None

        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.n_groups = n_groups
        self.batch_size = batch_size

        self.image_shape = image_shape

    @property
    def image_shape(self):
        """Return image shape."""
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value):
        """Set image shape."""
        if not value or self._image_shape == value:
            return
        self._image_shape = value

        h, w, n_filters = self.filter_shape
        conv_filter_shape = (n_filters, self.image_shape[2]/self.n_groups,
                             h, w)

        n_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
        W_value = np.asarray(
            np.random.normal(
                loc=0.,
                scale=np.sqrt(1. / n_out),
                size=conv_filter_shape
            ),
            dtype=theano.config.floatX
        )
        self.W_shared = theano.shared(W_value, borrow=True)

        b_values = np.zeros((n_filters,), dtype=theano.config.floatX)
        self.b_shared = theano.shared(b_values, borrow=True)

        self.params = [self.W_shared, self.b_shared]

    @property
    def input_shape(self):
        return self.image_shape

    @input_shape.setter
    def input_shape(self, value):
        self.image_shape = value

    @property
    def output_shape(self):
        """Return output shape."""
        image_h, image_w, n_channels = self.image_shape
        pad_h, pad_w = self.padding
        image_h += 2 * pad_h
        image_w += 2 * pad_w
        filter_h, filter_w, n_filters = self.filter_shape
        stride_h, stride_w = self.stride

        output_h = (image_h - filter_h) / stride_h + 1
        output_w = (image_w - filter_w) / stride_w + 1
        return (output_h, output_w, n_filters)

    def _reshape_input(self, raw_layer_input):
        """Return input in the correct format for convolutional layer.

        raw_layer_input: Input in the format (batch size, number of channels,
                                              image height, image width) or
                         compatible.
        """
        h, w, n_channels = self.image_shape
        conv_image_shape = (self.batch_size, n_channels, h, w)
        reshaped_input = raw_layer_input.reshape(conv_image_shape)

        pad_h, pad_w = self.padding
        h_in = h + 2*pad_h
        w_in = w + 2*pad_w
        shape = (self.batch_size, n_channels, h_in, w_in)

        val = np.zeros(shape=shape, dtype=theano.config.floatX)
        extra_pixels = T.alloc(val, shape[0], shape[1], shape[2], shape[3])
        extra_pixels = T.set_subtensor(
            extra_pixels[:, :, pad_h:pad_h+h, pad_w:pad_w+w], reshaped_input)
        return extra_pixels

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        n_channels = self.image_shape[2]
        n_filters = self.filter_shape[2]

        n_group_channels = n_channels / self.n_groups
        n_group_filters = n_filters / self.n_groups

        h, w = self.image_shape[0:2]
        pad_h, pad_w = self.padding
        group_image_shape = (self.batch_size, n_group_channels,
                             h + 2*pad_h, w + 2*pad_w)

        h, w = self.filter_shape[0:2]
        group_filter_shape = (n_group_filters, n_group_channels, h, w)

        conv_outputs = []
        for i in xrange(self.n_groups):
            conv_outputs += [conv.conv2d(
                input=self.input[:, i*n_group_channels:(i+1)*n_group_channels,
                                 :, :],
                filters=self.W_shared[i*n_group_filters:(i+1)*n_group_filters,
                                      :, :, :],
                filter_shape=group_filter_shape,
                image_shape=group_image_shape,
                subsample=self.stride
            )]

        conv_output = T.concatenate(conv_outputs, axis=1)
        return conv_output + self.b_shared.dimshuffle('x', 0, 'x', 'x')
