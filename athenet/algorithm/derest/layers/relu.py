from athenet.algorithm.derest.layers.layer import DerestLayer
from athenet.algorithm.numlike import assert_numlike


class DerestReluLayer(DerestLayer):

    def count_activation(self, input):
        return a_relu(input)

    def count_derivatives(self, output, input_shape):
        assert self.activation is not None
        return d_relu(output, self.activation)


def a_relu(layer_input):
    """Returns estimated activation of relu layer.

    :param Numlike layer_input: input
    :rtype: Numlike
    """
    assert_numlike(layer_input)
    try:
        res = layer_input.op_relu()
    except NotImplementedError:
        res = (layer_input + layer_input.abs()) * 0.5
    return res


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
