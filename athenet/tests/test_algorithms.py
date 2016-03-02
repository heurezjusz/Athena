import theano
import theano.tensor as T
import numpy as np
from unittest import TestCase, main
from copy import deepcopy

from athenet import Network
from athenet.layers import ConvolutionalLayer, Softmax, FullyConnectedLayer, \
    ReLU, MaxPool
from athenet.utils import DataLoader

from athenet.algorithm import simple_neuron_deleter, simple_neuron_deleter2
from athenet.algorithm.utils import list_of_percentage_rows_table, delete_row,\
    list_of_percentage_rows, list_of_percentage_columns, delete_column


def get_simple_network():
    result = Network([
        ConvolutionalLayer((5, 5, 30), image_shape=(30, 30, 1)),
        ReLU(),
        MaxPool((2, 2)),
        ConvolutionalLayer((5, 5, 20)),
        ReLU(),
        MaxPool((2, 2)),
        FullyConnectedLayer(100),
        ReLU(),
        FullyConnectedLayer(75),
        ReLU(),
        FullyConnectedLayer(50),
        ReLU(),
        FullyConnectedLayer(10)
    ])
    return result


class TestUtils(TestCase):
    def test_listing_table(self):
        table = np.asarray(
            [[1, 0, 0],
             [0, 0, 0]])
        rows = list_of_percentage_rows_table(table, 5)
        self.assertEqual(rows, [(1., 0, 5), (0., 1, 5)])

        table = np.asarray([[1], [0], [2], [0.5], [1.5]])
        rows = list_of_percentage_rows_table(table, 3)
        expected_rows = [(0.2, 0, 3), (0., 1, 3), (0.4, 2, 3), (0.1, 3, 3),
                         (0.3, 4, 3)]
        eps = 1e-9
        for (val, i, l_id), (e_val, e_i, e_l_id) in zip(rows, expected_rows):
            self.assertEqual((i, l_id), (e_i, e_l_id))
            self.assertTrue(abs(val - e_val) < eps)

    def test_listing_layers(self):
        net = get_simple_network()
        layer1 = net.weighted_layers[3]
        layer2 = net.weighted_layers[4]
        col1 = list_of_percentage_columns(1, layer1)
        row2 = list_of_percentage_rows(2, layer2)
        self.assertEqual(len(col1), len(row2))

        i = 0
        for (rval, ri, rl_id), (cval, ci, cl_id) in zip(row2, col1):
            self.assertEqual(ri, i)
            self.assertEqual(ci, i)
            self.assertEqual(rl_id, 2)
            self.assertEqual(cl_id, 1)
            i += 1

    def test_deleting(self):
        net = get_simple_network()

        layer = net.weighted_layers[3]
        layer.W = np.ones(layer.W.shape, dtype=theano.config.floatX)

        delete_row(layer, 42)
        delete_column(layer, 53)
        layer = net.weighted_layers[3]
        for r in xrange(layer.W.shape[0]):
            for c in xrange(layer.W.shape[1]):
                if c == 53 or r == 42:
                    self.assertEqual(layer.W[r][c], 0)
                else:
                    self.assertEqual(layer.W[r][c], 1)


def get_prepared_network():
    """return simple network filled with ones and with 200 in 3rd row 1st
       column of last layer
    """
    net = get_simple_network()
    for layer in net.weighted_layers:
        layer.W = np.ones(layer.W.shape, dtype=theano.config.floatX)
    W = net.weighted_layers[-1].W
    W[3][1] = 200
    net.weighted_layers[-1].W = W
    return net


def check_params(net, p, layer_limit):
    all_neurons = 0
    all_deleted = 0
    for layer in net.weighted_layers:
        if not isinstance(layer, FullyConnectedLayer):
            continue
        neurons = layer.W.shape[0]
        deleted = 0
        for r in xrange(layer.W.shape[0]):
            for c in xrange(layer.W.shape[1]):
                if layer.W[r][c] != 0:
                    break
            else:
                deleted += 1

        all_neurons += neurons
        all_deleted += deleted
        if deleted / 1. / neurons > layer_limit:
            return False

    return all_deleted / 1. / all_neurons <= p


class TestAlgorithms(TestCase):
    def test_sender(self):
        params = [((0.4, 0.5), (0.4, 0.5)),
                  ((1., 0.5), (0.5, 0.5)),
                  ((0.1, 0.2), (0.1, 0.2)),
                  ((0.7, 0.75), (0.7, 0.75)),
                  ((0.75, 0.6), (0.6, 0.6))]

        for config, (check_p, check_layer_limit) in params:
            net = get_prepared_network()
            simple_neuron_deleter(net, config)
            self.assertTrue(check_params(net, check_p, check_layer_limit))
            self.assertTrue(net.weighted_layers[-1].W[3][0] != 0)

    def test_sender2(self):
        params = [((0.4, 0.5), (0.4, 0.5)),
                  ((1., 0.5), (0.5, 0.5)),
                  ((0.1, 0.2), (0.1, 0.2)),
                  ((0.7, 0.75), (0.7, 0.75)),
                  ((0.75, 0.6), (0.6, 0.6))]

        for config, (check_p, check_layer_limit) in params:
            net = get_prepared_network()
            simple_neuron_deleter2(net, config)
            self.assertTrue(check_params(net, check_p, check_layer_limit))
            self.assertTrue(net.weighted_layers[-1].W[3][0] != 0)


if __name__ == '__main__':
    main(verbosity=2, catchbreak=True)
