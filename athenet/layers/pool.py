"""Pooling layer."""

from theano.tensor.signal import downsample

from athenet.layers import Layer


class MaxPool(Layer):
    """Max-pooling layer."""
    def __init__(self, poolsize, stride=None):
        """Create max-pooling layer.

        :poolsize: Pooling factor in the format (height, width).
        :stride: Pair representing interval at which to apply the filters.
        """
        super(MaxPool, self).__init__()
        self.poolsize = poolsize
        self.stride = stride

    @property
    def output_shape(self):
        image_h, image_w, n_channels = self.input_shape
        pool_h, pool_w = self.poolsize
        if self.stride:
            stride_h, stride_w = self.stride
        else:
            stride_h, stride_w = pool_h, pool_w

        output_h = (image_h - pool_h) / stride_h + 1
        output_w = (image_w - pool_w) / stride_w + 1
        return (output_h, output_w, n_channels)

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Input in the format (batch size, number of channels,
                                          image height, image width).
        :return: Layer output.
        """
        return downsample.max_pool_2d(
            input=layer_input,
            ds=self.poolsize,
            ignore_border=True,
            st=self.stride
        )
