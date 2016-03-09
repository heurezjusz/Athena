"""Data loader."""


class DataLoader(object):
    """Provides input and output data for a network.

    Base class for providing training, validation and testing data.
    Methods for loading and returning data should be implemented in all of its
    subclasses.
    """
    def __init__(self):
        self._batch_size = None
        self.n_train_batches = None
        self.n_val_batches = None
        self.n_test_batches = None

        self.train_set_size = 0
        self.val_set_size = 0
        self.test_set_size = 0
        self.train_data_available = False
        self.val_data_available = False
        self.test_data_available = False

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        self.n_train_batches = self.train_set_size / self.batch_size
        self.n_val_batches = self.val_set_size / self.batch_size
        self.n_test_batches = self.test_set_size / self.batch_size

    def _set_subset(self, data, value, index, size=1):
        data[index*self.batch_size:(index+size)*self.batch_size] = value

    def _get_subset(self, data, index, size=1):
        return data[index*self.batch_size:(index+size)*self.batch_size]

    def n_batches(self, data_type):
        """Return number of minibatches for given data type.

        :data_type: Instance of DataType.
        :return: Number of batches.
        """
        if data_type == 'training_data':
            return self.n_train_batches
        elif data_type == 'validation_data':
            return self.n_val_batches
        elif data_type == 'test_data':
            return self.n_test_batches
        return None

    def input(self, batch_index, data_type):
        """Return input data for given data type.

        :batch_index: Minibatch index.
        :data_type: Instance of DataType.
        :return: Input data.
        """
        if data_type == 'training_data':
            return self.train_input(batch_index)
        elif data_type == 'validation_data':
            return self.val_input(batch_index)
        elif data_type == 'test_data':
            return self.test_input(batch_index)
        return None

    def output(self, batch_index, data_type):
        """Return output data for given data type.

        :batch_index: Minibatch index.
        :data_type: Instance of DataType.
        :return: Output data.
        """
        if data_type == 'training_data':
            return self.train_output(batch_index)
        elif data_type == 'validation_data':
            return self.val_output(batch_index)
        elif data_type == 'test_data':
            return self.test_output(batch_index)
        return None

    def load_data(self, batch_index, data_type):
        """Load data for given data type.

        :batch_index: Minibatch index.
        :data_type: Instance of DataType.
        """
        if data_type == 'training_data':
            self.load_train_data(batch_index)
        elif data_type == 'validation_data':
            self.load_val_data(batch_index)
        elif data_type == 'test_data':
            self.load_test_data(batch_index)

    def train_input(self, batch_index):
        """Return minibatch of training data input.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of training data input.
        """
        raise NotImplementedError()

    def train_output(self, batch_index):
        """Return minibatch of training data output.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of training data output.
        """
        raise NotImplementedError()

    def val_input(self, batch_index):
        """Return minibatch of validation data input.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of validation data input.
        """
        raise NotImplementedError()

    def val_output(self, batch_index):
        """Return minibatch of validation data output.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of validation data output.
        """
        raise NotImplementedError()

    def test_input(self, batch_index):
        """Return minibatch of testing data input.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of testing data input.
        """
        raise NotImplementedError()

    def test_output(self, batch_index):
        """Return minibatch of testing data output.

        :batch_index: Symbolic variable representing minibatch index.
        :return: Minibatch of testing data output.
        """
        raise NotImplementedError()

    def load_train_data(self, batch_index):
        """Assure that train data are loaded into memory.

        :batch_index: Index of minibatch to be loaded.
        """
        pass

    def load_val_data(self, batch_index):
        """Assure that validation data are loaded into memory.

        :batch_index: Index of minibatch to be loaded.
        """
        pass

    def load_test_data(self, batch_index):
        """Assure that test data are loaded into memory.

        :batch_index: Index of minibatch to be loaded.
        """
        pass
