from athenet.algorithm.derest.layers import DerestLayer
from athenet.algorithm.numlike import assert_numlike


class DerestDropoutLayer(DerestLayer):

    def count_activation(self, layer_input):
        return self.a_dropout(layer_input, self.layer.p_dropout)

    def count_derivatives(self, layer_output, input_shape):
        return self.d_dropout(layer_output, self.layer.p_dropout)

    @staticmethod
    def a_dropout(layer_input, p_dropout):
        """Returns estimated activation of dropout layer.

        :param Numlike layer_input: input Numlike
        :param float p_dropout: probability of dropping in dropout
        :rtype: Numlike
        """
        assert_numlike(layer_input)
        return layer_input * (1.0 - p_dropout)

    @staticmethod
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