import unittest
import theano
import theano.tensor as T
import numpy as np
from athenet.layers import InceptionLayer, ConvolutionalLayer
from athenet.algorithm.derest.layers import get_derest_layer, DerestInceptionLayer


def eval_tensor_on_layers(layer1, layer2, tensor):
    layer1.input = T.tensor4()
    layer2.input_layer = layer1
    evaluate = theano.function(
        [layer1.input],
        layer2.output
    )
    return evaluate(tensor)


def prepare_inception_layer(image_size, n_filters):
    dummy_layer = ConvolutionalLayer((1,1,1))
    layer = InceptionLayer(n_filters=n_filters)
    input = np.ones(image_size)
    eval_tensor_on_layers(dummy_layer, layer, input)
    return layer


class TestInceptionLayer(unittest.TestCase):
    def _test_activation(self, image_size, n_filters):
        layer = prepare_inception_layer(image_size, n_filters)
        derest_layer = get_derest_layer(layer)
        self.assertIsInstance(derest_layer, DerestInceptionLayer)


    def test_activation(self):
        pass