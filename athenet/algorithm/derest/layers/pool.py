from athenet.algorithm.derest.layers import DerestLayer
from athenet.algorithm.numlike import assert_numlike
from athenet.algorithm.derest.utils import change_order


class DerestPoolLayer(DerestLayer):

    def count_activation(self, layer_input, normalize=False):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :return Numlike:
        """
        return a_pool(layer_input, change_order(self.layer.input_shape),
                      self.layer.poolsize, self.layer.stride,
                      self.layer.padding, self.layer.mode)

    def count_derivatives(self, layer_output, input_shape, normalize=False):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape:
        :return Numlike:
        """
        assert(self.activations is not None)
        return d_pool(layer_output, self.activations, input_shape,
                      self.layer.poolsize, self.layer.stride,
                      self.layer.padding, self.layer.mode)


def a_pool(layer_input, input_shp, poolsize, stride=(1, 1), padding=(0, 0),
           mode="max"):
    """Returns estimated activation of pool layer.

    :param Numlike layer_input: Numlike input in input_shp format
    :param tuple of 3 integers input_shp: input shape in format (n_channels,
                                          height, width)
    :param pair of integers poolsize: pool size in format (height, width)
    :param pair of integers stride: stride of max pool
    :param pair of integers padding: padding of pool, non-trivial padding is
                                     not allowed for 'max" mode
    :param 'max' or 'avg' mode: specifies whether it is max pool or average
                                pool
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    if mode not in ["max", "avg"]:
        raise ValueError("pool mode should be 'max' or 'avg'")
    is_max = mode == "max"
    # n_in, h, w - number of input channels, image height, image width
    n_in, h, w = input_shp
    n_out = n_in

    # padding
    pad_h, pad_w = padding
    if padding != (0, 0):
        layer_input = layer_input.reshape_for_padding(input_shp, padding)
        h += 2 * pad_h
        w += 2 * pad_w

    # fh, fw - pool height, pool width
    fh, fw = poolsize
    stride_h, stride_w = stride
    output_h = (h - fh) / stride_h + 1
    output_w = (w - fw) / stride_w + 1
    output_shp = (n_out, output_h, output_w)
    result = layer_input.from_shape(output_shp, neutral=True)
    for at_h in xrange(0, h - fh + 1, stride_h):
        # at_out_h - height of output corresponding to pool at position at_h
        at_out_h = at_h / stride_h
        for at_w in xrange(0, w - fw + 1, stride_w):
            # at_out_w - height of output corresponding to pool at
            # position at_w
            at_out_w = at_w / stride_w
            input_slice = layer_input[:, at_h:(at_h + fh), at_w:(at_w + fw)]
            if is_max:
                pool_res = input_slice.amax(axis=(1, 2), keepdims=False)
            else:
                pool_res = input_slice.sum(axis=(1, 2), keepdims=False) \
                    / float(fh * fw)
            result[:, at_out_h, at_out_w] = pool_res
    return result


def d_pool(output, activation, input_shape, poolsize, stride=(1, 1),
           padding=(0, 0), mode='max'):
    """Returns estimated impact of input of pool layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch size, number of channels,
                           height, width)
    :param Numlike activation: estimated activation of input
    :param input_shape: shape of layer input in format
                        (batch size, number of channels, height, width)
    :type input_shape: tuple of 4 integers
    :param pair of integers poolsize: pool size in format (height, width)
    :param pair of integers stride: stride of pool
    :param pair of integers padding: padding of pool
    :param 'max' or 'avg' mode: specifies whether it is max pool or average
                                pool
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """

    assert_numlike(activation)
    assert_numlike(output)
    if mode not in ['max', 'avg']:
        raise ValueError("pool mode should be 'max' or 'avg'")
    is_max = mode == 'max'
    if is_max:
        res = output.op_d_max_pool(activation, input_shape,
                                   poolsize, stride, padding)
    else:
        res = output.op_d_avg_pool(activation, input_shape,
                                   poolsize, stride, padding)
    return res
