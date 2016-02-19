"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer, ConvolutionalLayer
from athenet.data_loader import DataType
from athenet.utils import overwrite


class Network(object):
    """Neural network."""
    def __init__(self, layers):
        """Create neural network.

        layers: List of network's layers.
        """
        self._batch_size = None
        self._data_loader = None
        self.answers = None
        self.get_output = None

        self.verbosity = 1
        self._batch_index = T.lscalar()
        self._input = T.tensor4()
        self._correct_answers = T.ivector()
        self.layers = layers

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.batch_size = 1

    @property
    def data_loader(self):
        """Instance of class athenet.utils.DataLoader."""
        return self._data_loader

    @data_loader.setter
    def data_loader(self, value):
        self._data_loader = value
        if value:
            self.data_loader.batch_size = self.batch_size

    @property
    def batch_size(self):
        """Minibatch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if self._batch_size == value:
            return
        self._batch_size = value
        if self.data_loader:
            self.data_loader.batch_size = value

        for layer in self.convolutional_layers:
            layer.batch_size = self.batch_size
        self.layers[0].input = self._input
        for i in xrange(1, len(self.layers)):
            self.layers[i].input_layer = self.layers[i-1]

        output = self.layers[-1].output
        self.answers = T.argsort(-output, axis=1)
        self.get_output = theano.function(
            inputs=[self._input],
            outputs=[output.flatten(1), self.answers.flatten(1)]
        )

    def test_accuracy(self, top_range=1):
        """Return network's accuracy on the test data.

        :top_range: Number or list representing top ranges to be used.
                    Network's answer is considered correct if correct answer
                    is among top_range most probable answers given by the
                    network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, DataType.test_data)

    def val_accuracy(self, top_range=1):
        """Return network's accuracy on the validation data.

        :top_range: Number or list representing top ranges to be used.
                    Network's answer is considered correct if correct answer
                    is among top_range most probable answers given by the
                    network.
        :return: Number or list representing network accuracy for given top
                 ranges.
        """
        return self._get_accuracy(top_range, DataType.validation_data)

    def _get_accuracy(self, top_range, data_type):
        return_list = isinstance(top_range, list)
        if not return_list:
            top_range = [top_range]
        max_top_range = max(top_range)

        expanded = self._correct_answers.dimshuffle(0, 'x')
        expanded = expanded.repeat(max_top_range, axis=1)
        eq = T.eq(expanded, self.answers[:, :max_top_range])
        get_accuracy = theano.function(
            inputs=[self._batch_index],
            outputs=[T.any(eq[:, :top], axis=1).mean() for top in top_range],
            givens={
                self._input:
                    self.data_loader.input(self._batch_index, data_type),
                self._correct_answers:
                    self.data_loader.output(self._batch_index, data_type)
            }
        )

        n_batches = self.data_loader.n_batches(data_type)
        accuracy = np.zeros(shape=(n_batches, len(top_range)))
        interval = n_batches/10
        if interval == 0:
            interval = 1
        for batch_index in xrange(n_batches):
            self.data_loader.load_data(batch_index, data_type)
            accuracy[batch_index, :] = np.asarray(get_accuracy(batch_index))
            if self.verbosity >= 3 or (self.verbosity >= 2 and batch_index % interval == 0):
                partial_accuracy = accuracy[:batch_index+1, :].mean(axis=0)
                text = ''
                for a in partial_accuracy:
                    text += ' {:.2f}%'.format(100*a)
                overwrite('{}/{} minibatches accuracy:{}'.format(batch_index+1, n_batches, text))
        overwrite()

        accuracy = accuracy.mean(axis=0).tolist()
        if not return_list:
            return accuracy[0]
        return accuracy

    def get_params(self):
        """Return list of network's weights and biases.

        :return: List of pairs (W, b).
        """
        params = []
        for layer in self.weighted_layers:
            params += [(layer.W, layer.b)]
        return params

    def set_params(self, params):
        """Set network's weights and biases.

        :params: List of pairs (W, b).
        """
        for p, layer in zip(params, self.weighted_layers):
            layer.W = p[0]
            layer.b = p[1]

    def evaluate(self, net_input):
        """Return network output for a given input.

        Batch size must be equal 1 to use this method. If it isn't, it will be
        set to 1.

        :net_input: Input for the network.
        :return: A pair consisting of list of probabilities for every answer
                 index and list of answer indexes sorted by their
                 probabilities descending.
        """
        self.batch_size = 1
        net_input = np.asarray(net_input, dtype=theano.config.floatX)
        n_channels, height, width = net_input.shape
        net_input = np.resize(net_input, (1, n_channels, height, width))
        return self.get_output(net_input)

    def train(self, batch_size=None, n_epochs=100, learning_rate=0.1, momentum=0.):
        """Train and test the network.
 
        :batch_size: Size of minibatch to be set. If None then batch size that
                     is currenty set will be used.
        :n_epochs: Number of epochs.
        :learning_rate: Learning rate.
        :momentum: Momentum coefficient.   
        """
        if not self.data_loader:
            raise Exception('data loader is not set')
        if not self.data_loader.train_data_available:
            raise Exception('train data is not available')

        if batch_size is not None:
            self.batch_size = batch_size

        # set cost function for the last layer
        self.layers[-1].set_cost(self._correct_answers)
        cost = self.layers[-1].cost
        params = []
        for layer in self.weighted_layers:
            params += layer.params
        grad = T.grad(cost, params)

        if momentum:
            for layer in self.weighted_layers:
                layer.alloc_velocity()
            velocities = []
            for layer in self.weighted_layers:
                velocities += layer.params_velocity

            velocity_updates = [(v, momentum*v - learning_rate*derivative)
                                for v, derivative in zip(velocities, grad)]
            param_updates = [(param, param + v)
                             for param, v in zip(params, velocities)]
            updates = velocity_updates + param_updates
        else:
            updates = [(param, param - learning_rate*derivative)
                       for param, derivative in zip(params, grad)]

        train_model = theano.function(
            inputs=[self._batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self._input:
                    self.data_loader.train_input(self._batch_index),
                self._correct_answers:
                    self.data_loader.train_output(self._batch_index)
            },
        )

        val_interval = 2*self.data_loader.n_train_batches
        iteration = 0
        if self.verbosity >= 1:
            print 'Training with batch size = {}, learning rate = {}, '\
                  'momentum = {}'.format(self.batch_size, learning_rate,
                                         momentum)
            print '{} epochs, {} minibatches per epoch'\
                  .format(n_epochs, self.data_loader.n_train_batches)
        start_time = timeit.default_timer()
        for epoch in xrange(1, n_epochs+1):
            print 'Epoch {}'.format(epoch)
            epoch_start_time = timeit.default_timer()
            for batch_index in xrange(self.data_loader.n_train_batches):
                self.data_loader.load_train_data(batch_index)
                train_model(batch_index)
                if self.data_loader.val_data_available:
                    iteration += 1
                    if iteration % val_interval == 0:
                        accuracy = self.val_accuracy()
                        print '\tAccuracy on validation data: {:.2f}%'.format(
                            100*accuracy)
            epoch_end_time = timeit.default_timer()
            print '\tTime: {:.1f}s'.format(epoch_end_time - epoch_start_time)
            if epoch % 2 == 0:
                learning_rate *= 0.1
        end_time = timeit.default_timer()
        print 'Training time: {:.1f}s'.format(end_time - start_time)

        if momentum:
            for layer in self.weighted_layers:
                layer.free_velocity()
