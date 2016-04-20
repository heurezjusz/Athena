from athenet.algorithm.derest.layers.layer import DerestLayer
from athenet.algorithm.numlike import assert_numlike


class DerestSoftmaxLayer(DerestLayer):

    def count_activation(self, input):
        assert a_softmax(input, self.layer.input_shape)

    def count_derivatives(self, output):
        return d_softmax(output)


def a_softmax(layer_input, input_shp):
    """Returns estimated activation of softmax layer.
    :param Numlike layer_input: input
    :param integer input_shp: shape of 1D input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_softmax(input_shp)
    except NotImplementedError:
        exponents = layer_input.exp()
        res = exponents / exponents.sum()
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
