"""Functions that for any neuron calculate its range of derivative of output
with respect to this neuron. Functions should be invoked from the end to the
beginning of the network.

Every estimated impact of tensor on output of network is stored with batches.
Every entity in batches store impact on different output of network.
"""

from athenet.algorithm.numlike import Numlike, assert_numlike

# TODO: All functions below will be implemented.


def d_conv(output, input_shape, filter_shape, weights,
           stride=(1, 1), padding=(0, 0), n_groups=1):
    """Returns estimated impact of input of convolutional layer on output of
    network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param input_shape: shape of layer input in the format
                        (number of batches,
                         number of input channels,
                         image height,
                         image width)
    :type input_shape: tuple of 4 integers
    :param filter_shape: filter shape in the format (number of output channels,
                                                     filter height,
                                                     filter width)
    :type filter_shape: tuple of 3 integers
    :param weights: Weights tensor in format (number of output channels,
                                              number of input channels,
                                              filter height,
                                              filter width)
    :type weights: numpy.ndarray or theano tensor
    :param stride: pair representing interval at which to apply the filters.
    :type stride: pair of integers
    :param padding: pair representing number of zero-valued pixels to add on
                    each side of the input.
    :type padding: pair of integers
    :param n_groups: number of groups input and output channels will be split
                     into, two channels are connected only if they belong to
                     the same group.
    :type n_groups: integer
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    res = output.op_d_conv(input_shape, filter_shape,
                           weights, stride, padding, n_groups)
    return res


def d_dropout(output, p_dropout):
    """Returns estimated impact of input of dropout layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param float p_dropout: probability of dropping in dropout
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    return output * (1.0 - p_dropout)


def d_fully_connected(output, weights, input_shape):
    """Returns estimated impact of input of fully connected layer on output of
    network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param weights: weights of fully connected layer in format (n_in, n_out)
    :type weights: 2D numpy.ndarray or theano.tensor
    :param input_shape: shape of fully connected layer input in any format.
    :type input_shape: tuple of integers
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    try:
        res = output.dot(weights.T)
    except NotImplementedError:
        res = (output * weights).sum(1)
    return res.reshape(input_shape)


def d_norm(output, activation, input_shape, local_range, k, alpha, beta):
    # TODO: all
    """Returns estimated impact of input of LRN layer on output of network.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :param Numlike activation: estimated activation of input
    :param input_shape: shape of layer input in format
                        (number of batches, number of channels, height, width)
    :type input_shape: tuple of 4 integers
    :param int local_range: Local channel range. Should be odd, otherwise it
                            will be incremented.
    :param float k: Additive constant.
    :param float alpha: The scaling parameter.
    :param float beta: The exponent.
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    res = output.op_d_norm(activation, input_shape, local_range, k, alpha,
                           beta)
    return res


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


def d_softmax(output):
    """Returns estimated impact of input of softmax layer on output of network.

    .. warning: Current implementation only consider softmax as the last layer.

    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(output)
    return output


def d_relu(output, activation):
    """Returns estimated impact of input of relu layer on output of network.

    :param Numlike activation: estimated activation of input
    :param Numlike output: estimated impact of output of layer on output
                           of network in shape (batch_size, number of channels,
                           height, width)
    :returns: Estimated impact of input on output of network
    :rtype: Numlike
    """
    assert_numlike(activation)
    assert_numlike(output)
    res = output.op_d_relu(activation)
    return res
