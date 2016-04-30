from athenet.algorithm.derest.layers import DerestSoftmaxLayer,\
    DerestReluLayer, DerestPoolLayer, DerestNormLayer, DerestLayer, \
    DerestFullyConnectedLayer, DerestConvolutionalLayer, DerestDropoutLayer
from athenet.layers import Softmax, ReLU, PoolingLayer, LRN, \
    ConvolutionalLayer, Dropout, FullyConnectedLayer, InceptionLayer
from athenet.algorithm.derest.utils import add_tuples, change_order


def get_derest_layer(layer, normalize=False):
    """
    Return derest layer on which we can count activations, derivatives
        and derest algorithm

    :param Layer layer: network's original layer
    :return DerestLayer: new better derest layer
    """
    if isinstance(layer, Softmax):
        return DerestSoftmaxLayer(layer)
    if isinstance(layer, ReLU):
        return DerestReluLayer(layer)
    if isinstance(layer, PoolingLayer):
        return DerestPoolLayer(layer)
    if isinstance(layer, LRN):
        return DerestNormLayer(layer)
    if isinstance(layer, ConvolutionalLayer):
        return DerestConvolutionalLayer(layer)
    if isinstance(layer, Dropout):
        return DerestDropoutLayer(layer)
    if isinstance(layer, FullyConnectedLayer):
        return DerestFullyConnectedLayer(layer)
    if isinstance(layer, InceptionLayer):
        return DerestInceptionLayer(layer)
    raise NotImplementedError


class DerestInceptionLayer(DerestLayer):

    def __init__(self, layer):
        super(DerestInceptionLayer, self).__init__(layer)
        self.derest_layer_lists = []
        for layer_list in self.layer.layer_lists:
            derest_layer_list = []
            for l in layer_list:
                derest_layer_list.append(get_derest_layer(l))
            self.derest_layer_lists.append(derest_layer_list)

    @staticmethod
    def _normalize(data):
        a = data.max(-data).amax()
        return data / a

    def count_activation(self, input):
        results = None

        for derest_layer_list in self.derest_layer_lists:
            inp = input
            for derest_layer in derest_layer_list:
                if self.normalize_activations:
                    inp = self._normalize(inp)
                derest_layer.activations = inp
                inp = derest_layer.count_activation(inp)

            if results is None:
                results = inp
            else:
                results = results.concat(inp)

        return [results]

    def count_derivatives(self, output, input_shape):
        output_list = []
        last = 0
        for layer in self.layer.top_layers:
            channels = layer.output_shape[2]
            output_list.append(output[:, last : (last + channels), ::])
            last += channels

        batches = input_shape[0]
        result = None
        result = []
        i = 0
        for output, derest_list in zip(output_list, self.derest_layer_lists):
            if i == 4:
                break
            out = output
            j = 0
            for derest_layer in reversed(derest_list):
                print derest_layer
                print derest_layer.layer.output_shape

                if self.normalize_derivatives:
                    out = self._normalize(out)
                derest_layer.derivatives = out
                local_input_shape = add_tuples(
                    batches, change_order(derest_layer.layer.input_shape))
                out = derest_layer.count_derivatives(out, local_input_shape)
                if i == 3 and j == 2:
                    return [out]
                j += 1
            result.append(out)
            print "\033[39msuch a bigga lista"

            #result.append(out)
            i += 1
            #if result is None:
            #    result = out
            #else:
            #    result = result + out

        return result

    def count_derest(self, f):
        results = []
        for derest_layer_list in self.derest_layer_lists:
            for derest_layer in derest_layer_list:
                results += derest_layer.count_derest(f)
        return results
