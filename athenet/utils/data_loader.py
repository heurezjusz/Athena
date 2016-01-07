"""Data loader."""


class DataLoader(object):
    """Data loader."""
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
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        self.n_train_batches = self.train_set_size / self.batch_size
        self.n_val_batches = self.val_set_size / self.batch_size
        self.n_test_batches = self.test_set_size / self.batch_size

    def _get_subset(self, data, index, size=1):
        return data[index*self.batch_size:
                    (index+size)*self.batch_size]

    def train_input(self, batch_index):
        """Return minibatch of training data input.

        batch_index: Batch index.
        return: Minibatch of training data input.
        """
        raise NotImplementedError()

    def train_output(self, batch_index):
        """Return minibatch of training data output.

        batch_index: Batch index.
        return: Minibatch of training data output.
        """
        raise NotImplementedError()

    def val_input(self, batch_index):
        """Return minibatch of validation data input.

        batch_index: Batch index.
        return: Minibatch of validation data input.
        """
        raise NotImplementedError()

    def val_output(self, batch_index):
        """Return minibatch of validation data output.

        batch_index: Batch index.
        return: Minibatch of validation data output.
        """
        raise NotImplementedError()

    def test_input(self, batch_index):
        """Return minibatch of testing data input.

        batch_index: Batch index.
        return: Minibatch of testing data input.
        """
        raise NotImplementedError()

    def test_output(self, batch_index):
        """Return minibatch of testing data output.

        batch_index: Batch index.
        return: Minibatch of testing data output.
        """
        raise NotImplementedError()

    def load_train_data(self, batch_index):
        """Assure that train data are loaded into memory.

        batch_index: Batch index to be loaded.
        """
        pass

    def load_val_data(self, batch_index):
        """Assure that validation data are loaded into memory.

        batch_index: Batch index to be loaded.
        """
        pass

    def load_test_data(self, batch_index):
        """Assure that test data are loaded into memory.

        batch_index: Batch index to be loaded.
        """
        pass
