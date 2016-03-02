"""Buffer for storing large network data."""

import numpy as np

import theano


class Buffer(object):
    """Buffer storing data from contiguous subsequence of minibatches.

    Content of a buffer is a 4-dimensional floating-point tensor.
    """
    def __init__(self, data_loader):
        """Create data Buffer.

        :data_loader: Instance of DataLoader that will be using Buffer.
        """
        self.begin = -1
        self.end = 0
        self._offset = theano.shared(0)
        self.parent = data_loader

        # Create a 4-dimensinal tensor shared variable for data. Exact size of
        # the tensor is determined when data is set, and can change over time.
        self._data = theano.shared(
            np.zeros((1, 1, 1, 1), dtype=theano.config.floatX),
            borrow=True)

    @property
    def data(self):
        """Shared variable representing data stored in a buffer."""
        return self._data

    def __getitem__(self, key):
        """Return minibatches of given indices.

        Return data is taken from data array, however key represents
        minibatch index, not direct index in data array. Effectively, buffer
        can be used as if it contained all of the minibatches data.

        :key: Symbolic index or slice representing indices of minibatches to
              return.
        :return: Minibatches data.
        """
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return self._data[start-self._offset:stop-self._offset:step]
        else:
            return self._data[key-self._offset]

    def __setitem__(self, key, value):
        """Set buffer data.

        :key: Indices of minibatches to be stored.
        :value: Data to be stored in a buffer.
        """
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if step is not None and step != 1:
                raise NotImplementedError('step must be equal 1')
        else:
            start = key
            stop = start + self.parent.batch_size

        self.begin = start / self.parent.batch_size
        self.end = stop / self.parent.batch_size
        self._offset.set_value(start)
        self._data.set_value(value, borrow=True)

    def contains(self, batch_index):
        """Check if minibatch is contained in a buffer.

        :batch_index: Index of a minibatch.
        :return: True, if minibatch of a given index is contained in a buffer.
                 False otherwise.
        """
        return batch_index >= self.begin and batch_index < self.end
