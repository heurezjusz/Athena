from athenet.algorithm.derest.layers import DerestLayer
from athenet.algorithm.numlike import assert_numlike
from athenet.algorithm.derest.utils import change_order


class DerestNormLayer(DerestLayer):

    def count_activation(self, layer_input, normalize=False):
        """
        Returns estimated activations

        :param Numlike layer_input:
        :return Numlike:
        """
        return a_norm(layer_input, change_order(self.layer.input_shape),
                      self.layer.local_range, self.layer.k,
                      self.layer.alpha, self.layer.beta)

    def count_derivatives(self, layer_output, input_shape, normalize=False):
        """
        Returns estimated impact of input of layer on output of network

        :param Numlike layer_output:
        :param tuple input_shape:
        :return Numlike:
        """
        assert(self.activations is not None)
        return d_norm(layer_output, self.activations, input_shape,
                      self.layer.local_range, self.layer.k,
                      self.layer.alpha, self.layer.beta)


def a_norm(layer_input, input_shape, local_range=5, k=1, alpha=0.0001,
           beta=0.75):
    """Returns estimated activation of LRN layer.

    :param Numlike layer_input: Numlike input
    :param input_shape: shape of Interval in format (n_channels, height, width)
    :param integer local_range: size of local range in local range
                                normalization
    :param integer k: local range normalization k argument
    :param integer alpha: local range normalization alpha argument
    :param integer beta: local range normalization beta argument
    :type input_shape: tuple of 3 integers
    :rtype: Numlike
    """

    assert_numlike(layer_input)
    try:
        return layer_input.op_norm(input_shape, local_range, k, alpha, beta)
    except NotImplementedError:
        half = local_range / 2
        sq = layer_input.square()
        n_channels, h, w = input_shape
        extra_channels = layer_input.from_shape((n_channels + 2 * half, h, w),
                                                neutral=True)
        extra_channels[half:half + n_channels, :, :] = sq
        local_sums = layer_input.from_shape(input_shape, neutral=True)

        for i in xrange(local_range):
            local_sums += extra_channels[i:i + n_channels, :, :]

        return layer_input /\
            ((local_sums * (alpha / local_range) + k).power(beta))


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
