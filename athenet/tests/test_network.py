""" Tests for network """

import theano
import theano.tensor as T
import numpy as np
from unittest import TestCase, main

from athenet import Network
from athenet.layers import ConvolutionalLayer, Softmax, FullyConnectedLayer, \
    Activation, ReLU
from athenet.data_loader import DataLoader


class TestNetworkBasics(TestCase):
    def test_layers_lists(self):
        def foo(x):
            return T.minimum(x, 0.)

        net = Network([
            ConvolutionalLayer(image_shape=(42, 21, 2),
                               filter_shape=(5, 5, 3)),
            ReLU(),
            ConvolutionalLayer(filter_shape=(5, 5, 3)),
            Activation(foo),
            FullyConnectedLayer(10),
            ReLU(),
            Softmax(),
            FullyConnectedLayer(3),
        ])

        self.assertEqual(len(net.convolutional_layers), 2)
        self.assertEqual(len(net.weighted_layers), 4)
        self.assertEqual(len(net.layers), 8)


class DummyDataLoader(DataLoader):
    def __init__(self, data_in, data_out):
        """ data_in: list of batches witch input data
            data_out: list of batches witch output data
        """
        super(DummyDataLoader, self).__init__()
        self.data_in = theano.shared(np.asarray((data_in),
                                                dtype=theano.config.floatX))
        self.data_out = T.cast(theano.shared(np.asarray(data_out)), 'int32')

        self.train_set_size = len(data_in)
        self.val_set_size = len(data_in)
        self.test_set_size = len(data_in)

        self.train_data_available = True
        self.val_data_available = False
        self.test_data_available = True

    def train_input(self, batch_index):
        return self.data_in[batch_index:batch_index + 1]

    def train_output(self, batch_index):
        return self._get_subset(self.data_out, batch_index)[0]

    def test_input(self, batch_index):
        return self.train_input(batch_index)

    def test_output(self, batch_index):
        return self.train_output(batch_index)


class TestTrainingProcess(TestCase):
    def test_training_process(self):
        net = Network([FullyConnectedLayer(n_in=3, n_out=6),
                       ReLU(),
                       FullyConnectedLayer(n_out=2),
                       Softmax()])
        self.assertEqual(net.batch_size, 1)
        data_in = []
        data_out = []
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    data_in.append([[[1. * i, 1. * j, 1. * k]]])
                    data_out.append([i ^ j ^ k])
        net.data_loader = DummyDataLoader(data_in, data_out)
        net.train(n_epochs=500, batch_size=1)

        net.batch_size = 3
        correct = 0.
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    raw = net.evaluate([[[i, j, k]]])[0]
                    out = net.evaluate([[[i, j, k]]])[1]
                    self.assertTrue(raw[out[0]] > raw[out[1]])
                    if out[0] == i ^ j ^ k:
                        correct += 1.

        self.assertEqual(correct / len(data_in), net.test_accuracy())
        # it should be automatically corrected
        self.assertEqual(net.batch_size, 1)


if __name__ == '__main__':
    main(verbosity=2, catchbreak=True)
