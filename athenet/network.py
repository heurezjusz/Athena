"""Main Network class."""

import timeit
import numpy as np

import theano
import theano.tensor as T

from athenet.layers import WeightedLayer, ConvolutionalLayer


class Network(object):
    """Neural network."""

    # Early stopping parameters
    initial_patience = 10000
    patience_increase = 2

    def __init__(self, layers, batch_size=1):
        """Create neural network.

        layers: List of network's layers
        batch_size: Minibatch size
        """
        self._data_accuracy = None
        self._test_data_accuracy = None
        self._val_data_accuracy = None
        self._batch_size = None
        self._data_loader = None

        self.output = None
        self.train_output = None
        self.get_output = None

        self.layers = layers
        self.x = T.tensor4('x')
        self.y = T.ivector('y')
        self._batch_index = T.lscalar()

        self.weighted_layers = [layer for layer in self.layers
                                if isinstance(layer, WeightedLayer)]
        self.convolutional_layers = [layer for layer in self.weighted_layers
                                     if isinstance(layer, ConvolutionalLayer)]

        self.batch_size = batch_size

    @property
    def data_loader(self):
        """Return data loader."""
        return self._data_loader

    @data_loader.setter
    def data_loader(self, value):
        """Set data loader."""
        self._data_loader = value
        self.data_loader.batch_size = self.batch_size
        self._update()

    @property
    def batch_size(self):
        """Return batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set batch size."""
        if self._batch_size == value:
            return

        self._batch_size = value
        if self.data_loader:
            self.data_loader.batch_size = value

        for layer in self.convolutional_layers:
            layer.batch_size = self.batch_size
        self.layers[0].input = self.x
        for i in xrange(1, len(self.layers)):
            self.layers[i].input_layer = self.layers[i-1]

        self.output = self.layers[-1].output
        self.train_output = self.layers[-1].train_output
        y_out = T.argmax(self.output, axis=1)

        self.params = []
        for layer in self.weighted_layers:
            self.params += layer.params

        self._data_accuracy = T.mean(T.eq(self.y, y_out))
        self.get_output = theano.function(
            inputs=[self.x],
            outputs=self.output.flatten(1)
        )

        if self.data_loader:
            self._update()

    def test_accuracy(self):
        """Return average network accuracy on the test data.

        Datasets must be set before using this method.

        return: A number between 0 and 1 representing average accuracy
        """
        test_accuracies = [self._test_data_accuracy(i) for i in
                           xrange(self.data_loader.n_test_batches)]
        return np.mean(test_accuracies)

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

    def evaluate(self, x_in):
        """Return network output for a given input.

        x_in: Input for the network
        """
        batch_size = self.batch_size
        self.batch_size = 1
        x_in = np.asarray(x_in, dtype=theano.config.floatX)
        n_channels, height, width = x_in.shape
        x_in = np.resize(x_in, (1, n_channels, height, width))
        self.batch_size = batch_size
        return self.get_output(x_in)

    def train(self, learning_rate=0.1, n_epochs=100, batch_size=None):
        """Train and test the network.

        learning_rate: Learning rate
        n_epochs: Number of epochs
        batch_size: Size of minibatch
        """
        if not self.data_loader:
            raise Exception('Data loader is not set')
        if not self.data_loader.test_data_available:
            raise Exception('Test data are not available')

        old_batch_size = self.batch_size
        if batch_size:
            self.batch_size = batch_size

        # set cost function for the last layer
        self.layers[-1].set_cost(self.y)
        cost = self.layers[-1].cost

        grad = T.grad(cost, self.params)
        updates = [(param, param - learning_rate*derivative)
                   for param, derivative in zip(self.params, grad)]

        train_model = theano.function(
            inputs=[self._batch_index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: self.data_loader.train_input(self._batch_index),
                self.y: self.data_loader.train_output(self._batch_index)
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
            for minibatch_index in xrange(self.data_loader.n_train_batches):
                train_model(minibatch_index)
                if self.data_loader.val_data_available:
                    iteration += 1
                    if iteration % val_interval == 0:
                        val_accuracies = [
                            self._val_data_accuracy(i)
                            for i in xrange(self.data_loader.n_val_batches)]
                        val_accuracy = np.mean(val_accuracies)
                        print '\tAccuracy on validation data: {:.2f}%'.format(
                            100 * val_accuracy)
                        if val_accuracy > best_val_accuracy:
                            patience = max(patience, iteration *
                                           self.patience_increase)
                            best_val_accuracy = val_accuracy

                if patience <= iteration:
                    done_looping = True
                    break
        end_time = timeit.default_timer()

        print 'Training time: {:.1f}s'.format(end_time - start_time)
        if self.data_loader.test_data_available:
            print 'Accuracy on test data: {:.2f}%'.format(
                100*self.test_accuracy())

        self.batch_size = batch_size

    def _update(self):
        """Update fields that depend on both batch size and data loader."""
        if self.data_loader.val_data_available:
            self._val_data_accuracy = theano.function(
                inputs=[self._batch_index],
                outputs=self._data_accuracy,
                givens={
                    self.x: self.data_loader.val_input(self._batch_index),
                    self.y: self.data_loader.val_output(self._batch_index)
                }
            )
        else:
            self._val_data_accuracy = None

        if self.data_loader.test_data_available:
            self._test_data_accuracy = theano.function(
                inputs=[self._batch_index],
                outputs=self._data_accuracy,
                givens={
                    self.x: self.data_loader.test_input(self._batch_index),
                    self.y: self.data_loader.test_output(self._batch_index)
                }
            )
        else:
            self._test_data_accuracy = None
