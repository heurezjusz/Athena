"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer, ConvolutionalLayer


class Network(object):
    """Neural network."""

    verbosity = 0

    # Early stopping parameters
    initial_patience = 10000
    patience_increase = 2

    def __init__(self, layers):
        """Create neural network.

        layers: List of network's layers.
        """
        self._data_accuracy = None
        self._test_data_accuracy = None
        self._val_data_accuracy = None
        self._batch_size = None
        self._data_loader = None

        self.params = None
        self.output = None
        self.answer = None
        self.train_output = None
        self.get_output = None

        self._batch_index = T.lscalar()
        self._top_range = T.iscalar()

        self.layers = layers
        self.input = T.tensor4()
        self.correct_answer = T.ivector()

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        # batch_size: Minibatch size
        self.batch_size = 1
        # data_loader: instance of class athenet.utils.DataLoader
        self.data_loader = None

    @property
    def data_loader(self):
        return self._data_loader

    @data_loader.setter
    def data_loader(self, value):
        if not value:
            return
        self._data_loader = value
        self.data_loader.batch_size = self.batch_size
        self._update()

    @property
    def batch_size(self):
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
        self.layers[0].input = self.input
        for i in xrange(1, len(self.layers)):
            self.layers[i].input_layer = self.layers[i-1]

        self.params = []
        for layer in self.weighted_layers:
            self.params += layer.params

        self.output = self.layers[-1].output
        self.train_output = self.layers[-1].train_output
        self.answer = T.argsort(-self.output, axis=1)

        expanded = self.correct_answer.dimshuffle(0, 'x')
        expanded = expanded.repeat(self._top_range, axis=1)
        eq = T.eq(expanded, self.answer[:, :self._top_range])
        self._data_accuracy = T.any(eq, axis=1).mean()

        self.get_output = theano.function(
            inputs=[self.input],
            outputs=[self.output.flatten(1), self.answer.flatten(1)]
        )

        self._update()

    def _get_accuracy(self, top_range, accuracy_fun, load_fun, n_batches):
        return_list = isinstance(top_range, list)
        if not return_list:
            top_range = [top_range]

        interval = n_batches / 10
        if interval == 0:
            interval = 1
        accuracies = []
        for top in top_range:
            batch_accuracies = []
            for batch_index in xrange(n_batches):
                load_fun(batch_index)
                accuracy = accuracy_fun(batch_index, top)
                batch_accuracies += [accuracy]
                if self.verbosity >= 2 or\
                    (self.verbosity >= 1 and batch_index % interval == 0):
                    print 'Minibatch {} top-{} accuracy: {:.1f}%'.format(
                        batch_index, top, 100*accuracy)
            accuracies += [np.mean(batch_accuracies)]

        if not return_list:
            return accuracies[0]
        return accuracies

    def test_accuracy(self, top_range=1):
        """Return network's accuracy on the test data.

        top_range: Number or list represinting top ranges to be used.
                   Network's answer is considered correct if correct answer is
                   among top_range most probable answers given by network.
        return: Number or list representing network accuracy for given top
                ranges.
        """
        return self._get_accuracy(top_range,
                                  self._test_data_accuracy,
                                  self.data_loader.load_test_data,
                                  self.data_loader.n_test_batches)

    def val_accuracy(self, top_range=1):
        """Return network's accuracy on the validation data.

        top_range: Number or list represinting top ranges to be used.
                   Network's answer is considered correct if correct answer is
                   among top_range most probable answers given by network.
        return: Number or list representing network accuracy for given top
                ranges.
        """
        return self._get_accuracy(top_range,
                                  self._val_data_accuracy,
                                  self.data_loader.load_val_data,
                                  self.data_loader.n_val_batches)

    def get_params(self):
        """Return network's weights and biases.

        return: List of tuples (W, b)
        """
        params = []
        for layer in self.weighted_layers:
            params += [(layer.W, layer.b)]
        return params

    def set_params(self, params):
        """Set network's weights and biases.

        params: List of tuples (W, b)
        """
        for p, layer in zip(params, self.weighted_layers):
            layer.W = p[0]
            layer.b = p[1]

    def evaluate(self, net_input):
        """Return network output for a given input.

        Batch size should be set to 1 before using this method. If it isn't,
        it will be set to 1.

        net_input: Input for the network
        """
        self.batch_size = 1
        net_input = np.asarray(net_input, dtype=theano.config.floatX)
        n_channels, height, width = net_input.shape
        net_input = np.resize(net_input, (1, n_channels, height, width))
        return self.get_output(net_input)

    def train(self, learning_rate=0.1, n_epochs=100, batch_size=None):
        """Train and test the network.

        learning_rate: Learning rate.
        n_epochs: Number of epochs.
        batch_size: Size of minibatch to be set. If None then batch size that
                    is currenty set will be used.
        """
        if not self.data_loader:
            raise Exception('Data loader is not set')
        if not self.data_loader.test_data_available:
            raise Exception('Test data are not available')

        self.batch_size = batch_size

        # set cost function for the last layer
        self.layers[-1].set_cost(self.correct_answer)
        cost = self.layers[-1].cost

        grad = T.grad(cost, self.params)
        updates = [(param, param - learning_rate*derivative)
                   for param, derivative in zip(self.params, grad)]

        train_model = theano.function(
            inputs=[self._batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: self.data_loader.train_input(self._batch_index),
                self.correct_answer:
                    self.data_loader.train_output(self._batch_index)
            }
        )

        patience = self.initial_patience
        val_interval = min(self.data_loader.n_train_batches, patience/2)
        best_val_accuracy = 0.0
        epoch = 0
        iteration = 0
        done_looping = False

        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            print 'Epoch {}'.format(epoch)
            for batch_index in xrange(self.data_loader.n_train_batches):
                self.data_loader.load_train_data(batch_index)
                train_model(batch_index)
                if self.data_loader.val_data_available:
                    iteration += 1
                    if iteration % val_interval == 0:
                        accuracy = self.val_accuracy()
                        print '\tAccuracy on validation data: {:.2f}%'.format(
                            100*accuracy)
                        if accuracy > best_val_accuracy:
                            patience = max(patience,
                                           iteration*self.patience_increase)
                            best_val_accuracy = accuracy

                if patience <= iteration:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        print 'Training time: {:.1f}s'.format(end_time - start_time)
        if self.data_loader.test_data_available:
            print 'Accuracy on test data: {:.2f}%'.format(
                100*self.test_accuracy())

    def _update(self):
        """Update fields that depend on both batch size and data loader."""
        if not self.data_loader:
            return

        if self.data_loader.val_data_available:
            self._val_data_accuracy = theano.function(
                inputs=[self._batch_index, self._top_range],
                outputs=self._data_accuracy,
                givens={
                    self.input:
                        self.data_loader.val_input(self._batch_index),
                    self.correct_answer:
                        self.data_loader.val_output(self._batch_index)
                }
            )
        else:
            self._val_data_accuracy = None

        if self.data_loader.test_data_available:
            self._test_data_accuracy = theano.function(
                inputs=[self._batch_index, self._top_range],
                outputs=self._data_accuracy,
                givens={
                    self.input:
                        self.data_loader.test_input(self._batch_index),
                    self.correct_answer:
                        self.data_loader.test_output(self._batch_index)
                }
            )
        else:
            self._test_data_accuracy = None
