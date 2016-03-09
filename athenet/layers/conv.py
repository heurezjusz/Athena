"""Convolutional layer."""

import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer
from athenet.utils import cudnn_available


class ConvolutionalLayer(WeightedLayer):
    """Convolutional layer."""
    def __init__(self, filter_shape, image_shape=None, stride=(1, 1),
                 padding=(0, 0), n_groups=1, input_layer_name=None,
                 name='conv'):
        """Create convolutional layer.

        :filter_shape: Filter shape in the format
                       (filter height, filter width, number of filters).
        :image_shape: Image shape in the format
                      (image height, image width, number of channels).
        :stride: Pair representing interval at which to apply the filters.
        :padding: Pair representing number of zero-valued pixels to add on
                  each side of the input.
        :n_groups: Number of groups input and output channels will be split
                   into. Two channels are connected only if they belong to the
                   same group.
        """
        super(ConvolutionalLayer, self).__init__(input_layer_name, name)
        self._image_shape = None

        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.n_groups = n_groups
        self.image_shape = image_shape

    @property
    def image_shape(self):
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value):
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

        b_value = np.zeros((n_filters,), dtype=theano.config.floatX)
        self.b_shared = theano.shared(b_value, borrow=True)

    @property
    def input_shape(self):
        return self.image_shape

    @input_shape.setter
    def input_shape(self, value):
        self.image_shape = value

    @property
    def output_shape(self):
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

        :raw_layer_input: Input in the format (batch size, number of channels,
                                               image height, image width) or
                          compatible.
        """
        if self.padding == (0, 0):
            return raw_layer_input

        h, w, n_channels = self.image_shape
        pad_h, pad_w = self.padding
        h_in = h + 2*pad_h
        w_in = w + 2*pad_w

        extra_pixels = T.alloc(
            np.array(0., dtype=theano.config.floatX),
            raw_layer_input.shape[0], n_channels, h_in, w_in)
        extra_pixels = T.set_subtensor(
            extra_pixels[:, :, pad_h:pad_h+h, pad_w:pad_w+w], raw_layer_input)
        return extra_pixels

    def _get_output(self, layer_input):
        n_channels = self.image_shape[2]
        n_filters = self.filter_shape[2]

        n_group_channels = n_channels / self.n_groups
        n_group_filters = n_filters / self.n_groups

        # By default, Theano doesn't use cuDNN convolutions if
        # subsample != (1, 1), so we need to call it manually
        if cudnn_available():  # use cuDNN convolutions
            conv_outputs = [theano.sandbox.cuda.dnn.dnn_conv(
                img=self.input[:, i*n_group_channels:(i+1)*n_group_channels,
                               :, :],
                kerns=self.W_shared[i*n_group_filters:(i+1)*n_group_filters,
                                    :, :, :],
                subsample=self.stride
            ) for i in xrange(self.n_groups)]
        else:  # let Theano decide which implementation to use
            h, w = self.image_shape[0:2]
            pad_h, pad_w = self.padding
            if self.batch_size is not None:
                group_image_shape = (self.batch_size, n_group_channels,
                                     h + 2*pad_h, w + 2*pad_w)
            else:
                group_image_shape = None
            h, w = self.filter_shape[0:2]
            group_filter_shape = (n_group_filters, n_group_channels, h, w)

            conv_outputs = [theano.tensor.nnet.conv.conv2d(
                input=self.input[:, i*n_group_channels:(i+1)*n_group_channels,
                                 :, :],
                filters=self.W_shared[i*n_group_filters:(i+1)*n_group_filters,
                                      :, :, :],
                filter_shape=group_filter_shape,
                image_shape=group_image_shape,
                subsample=self.stride
            ) for i in xrange(self.n_groups)]

        conv_output = T.concatenate(conv_outputs, axis=1)
        return conv_output + self.b_shared.dimshuffle('x', 0, 'x', 'x')
