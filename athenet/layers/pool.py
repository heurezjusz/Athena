"""Pooling layer."""

from theano.tensor.signal import downsample

from athenet.layers import Layer


class PoolingLayer(Layer):
    """Pooling layer."""
    def __init__(self, poolsize, stride=None, mode='max'):
        """Create pooling layer.

        :poolsize: Shape of pooling filter in the format (height, width).
        :stride: Pair representing interval at which to apply the filters.
                 If None, then stride of the size of the pooling filter will be
                 used.
        :mode: Pooling method: 'max' or 'avg'. Default 'max'.
        """
        super(PoolingLayer, self).__init__()
        self.poolsize = poolsize
        if stride is None:
            self.stride = poolsize
        else:
            self.stride = stride
        self.mode = mode

    @property
    def output_shape(self):
        image_h, image_w, n_channels = self.input_shape
        pool_h, pool_w = self.poolsize
        if self.stride is not None:
            stride_h, stride_w = self.stride
        else:
            stride_h, stride_w = pool_h, pool_w

        output_h = (image_h - pool_h) / stride_h + 1
        output_w = (image_w - pool_w) / stride_w + 1
        return (output_h, output_w, n_channels)

    def _get_output(self, layer_input):
        """Return layer's output.

        :layer_input: Input in the format (batch size, number of channels,
                                           image height, image width).
        :return: Layer output.
        """
        if self.stride == self.poolsize:
            stride = None
        else:
            stride = self.stride
        if self.mode == 'avg':
            mode = 'average_exc_pad'
        else:
            mode = self.mode

        return downsample.max_pool_2d(
            input=layer_input,
            ds=self.poolsize,
            ignore_border=True,
            st=stride,
            mode=mode,
        )


class MaxPool(PoolingLayer):
    def __init__(self, poolsize, stride=None):
        """Create max-pooling layer."""
        super(MaxPool, self).__init__(poolsize, stride, 'max')


class AvgPool(PoolingLayer):
    def __init__(self, poolsize, stride=None):
        """Create average-pooling layer."""
        super(AvgPool, self).__init__(poolsize, stride, 'avg')
