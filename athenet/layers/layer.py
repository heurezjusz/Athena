"""Layer and WeightedLayer."""


class Layer(object):
    """Network layer."""
    def __init__(self):
        self.output = None
        self.train_output = None
        self.cost = None
        self._input_shape = None

        self._input = None
        self._train_input = None
        self._input_layer = None

    def _reshape_input(self, raw_layer_input):
        """Return input in the correct format for given layer.

        raw_layer_input: Layer input.
        """
        return raw_layer_input

    def _get_output(self, layer_input):
        """Return layer's output.

        layer_input: Layer input.
        """
        return layer_input

    def _get_train_output(self, layer_input):
        """Return layer's output used for training.

        layer_input: Layer input.
        """
        return self._get_output(layer_input)

    @property
    def input(self):
        """Return layer input."""
        return self._input

    @input.setter
    def input(self, value):
        """Set layer input."""
        self._input = self._reshape_input(value)
        self.output = self._get_output(self.input)

    @property
    def train_input(self):
        """Return layer input used for training."""
        if self._train_input:
            return self._train_input
        return self._input

    @train_input.setter
    def train_input(self, value):
        """Set layer input used for training."""
        self._train_input = self._reshape_input(value)
        self.train_output = self._get_train_output(self.train_input)

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = value

    @property
    def output_shape(self):
        """Return output shape."""
        return self.input_shape

    @property
    def input_layer(self):
        """Return input layer."""
        return self._input_layer

    @input_layer.setter
    def input_layer(self, input_layer):
        """Set input layer."""
        self._input_layer = input_layer
        self.input_shape = input_layer.output_shape

        self.input = input_layer.output
        self.train_input = input_layer.train_output


class WeightedLayer(Layer):
    """Layer with weights and biases."""
    def __init__(self):
        """Create weighted layer.

        weights: Array of weights's values
        biases: Array of biases' values
        """
        super(WeightedLayer, self).__init__()
        self.W_shared = None
        self.b_shared = None
        self.params = None

    @property
    def W(self):
        """Return copy of the layer's weights.

        return: Array of weights' values
        """
        return self.W_shared.get_value()

    @W.setter
    def W(self, value):
        """Set the layer's weights.

        value: Array of weights' values
        """
        self.W_shared.set_value(value)

    @property
    def b(self):
        """Return copy of the layer's biases.

        return: Array of biases' values
        """
        return self.b_shared.get_value()

    @b.setter
    def b(self, value):
        """Set the layer's biases.

       value: Array of biases' values
        """
        self.b_shared.set_value(value)
