"""Template class with arithmetic operations that can be passed through neural
network.

All classes that are being used for derest should inherit from this class."""


class Numlike(object):
    """Template class with arithmetic operations that can be passed through
    neural network.

    All classes that are being used for derest should inherit from this
    class."""

    def __init__(self):
        """Create numlike."""
        pass

    def __getitem__(self, at):
        """Returns specified slice of numlike.

        :at: Coordinates / slice to be taken.
        :rtype: Numlike
        """
        raise NotImplementedError

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.

        :at: Coordinates / slice to be set.
        :other: Data to be put at 'at'.
        """
        raise NotImplementedError

    @property
    def shape(self):
        """Returns shape of numlike.

        :rtype: integer or tuple of integers or theano shape
        """
        raise NotImplementedError

    def __add__(self, other):
        """Returns sum of two numlikes.

        :param other: value to be added.
        :type other: Numlike or np.ndarray or theano.tensor
        :rtype: Numlike
        """
        raise NotImplementedError

    def __sub__(self, other):
        """Returns difference between two numlikes.

        :param other: value to be subtracted.
        :type other: Numlike or np.ndarray or theano.tensor
        :rtype: Numlike
        """
        raise NotImplementedError

    def __mul__(self, other):
        """Returns product of two numlikes.

        :param other: value to be multiplied.
        :type other: Numlike or np.ndarray or theano.tensor
        :rtype: Numlike
        """
        raise NotImplementedError

    def __div__(self, other):
        """Returns quotient of self and other.

        :param other: divisor
        :type other: Numlike or np.ndarray or theano.tensor
        :rtype: Numlike
        """
        raise NotImplementedError

    def __rdiv__(self, other):
        """Returns quotient of other and self.

        :param other: dividend
        :type other: float
        :rtype: Nplike
        .. warning:: divisor (self) should not contain zero, other must be
                     float
        """
        raise NotImplementedError

    def reciprocal(self):
        """Returns reciprocal of the Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    def neg(self):
        """Returns (-1) * Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    def exp(self):
        """Returns Numlike representing the exponential of the Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    def square(self):
        """Returns square of the Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    def power(self, exponent):
        """For numlike N, returns N^exponent.

        :param float exponent: Number to be passed as exponent to N^exponent.
        :rtype: Numlike
        """
        raise NotImplementedError

    def dot(self, other):
        """Dot product of numlike vector and a other.

        :param unspecified other: second dot param, type to be specified
        :rtype: Numlike
        """
        raise NotImplementedError

    def max(self, other):
        """Returns maximum of self and other.

        :param unspecified other: second masx param, type to be specified
        :rtype: Numlike
        """
        raise NotImplementedError

    def amax(self, axis=None, keepdims=False):
        """Returns maximum of a Numlike along an axis.

        Works like theano.tensor.max

        :param axis: axis along which max is evaluated
        :param Boolean keepdims: whether flattened dimensions should remain
        :rtype: Numlike
        """
        raise NotImplementedError

    def reshape(self, shape):
        """Reshapes numlike tensor like theano Tensor.

        :param integer tuple shape: shape to be set
        :rtype: Numlike
        """
        raise NotImplementedError

    def flatten(self):
        """Flattens numlike tensor like theano Tensor.

        :rtype: Numlike
        """
        raise NotImplementedError

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Vector operation like in numpy.ndarray.

        :param axis: axis along which this function sums
        :param numeric type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        :type axis: integer, tuple of integers or None
        :rtype: Numlike

        """
        raise NotImplementedError

    def abs(self):
        """Returns absolute value of Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    @property
    def T(self):
        """Vector operation like in numpy.ndarray.

        :rtype: Numlike
        """
        raise NotImplementedError

    @staticmethod
    def from_shape(shp, neutral=True):
        """Returns Numlike of given shape.

        :param integer tuple shp: shape to be set
        :param Boolean neutral: whether created Numlike should have neutral
                        values or significant values.
        :rtype: Numlike
        """
        raise NotImplementedError

    def eval(self, *args):
        """Returns some readable form of stored value."""
        raise NotImplementedError

    def op_relu(self):
        """Returns result of relu operation on given Numlike.

        :rtype: Numlike
        """
        raise NotImplementedError

    def op_softmax(self, input_shp):
        """Returns result of softmax operation on given Numlike.

        :param integer input_shp: shape of 1D input
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_norm(self, input_shape, local_range, k, alpha, beta):
        """Returns estimated activation of LRN layer.

        :param input_shape: shape of input in format
                            (n_channels, height, width)
        :param integer local_range: size of local range in local range
                                    normalization
        :param integer k: local range normalization k argument
        :param integer alpha: local range normalization alpha argument
        :param integer beta: local range normalization beta argument
        :type input_shape: tuple of 3 integers
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_conv(self, weights, image_shape, filter_shape, biases, stride,
                padding, n_groups):
        """Returns estimated activation of convolution applied to Numlike.

        :param weights: weights tensor in format (number of output channels,
                                                  number of input channels,
                                                  filter height,
                                                  filter width)
        :param image_shape: shape of input in the format
                    (number of input channels, image height, image width)
        :param filter_shape: filter shape in the format
                             (number of output channels, filter height,
                              filter width)
        :param biases: biases in convolution
        :param stride: pair representing interval at which to apply the filters
        :param padding: pair representing number of zero-valued pixels to add
                        on each side of the input.
        :param n_groups: number of groups input and output channels will be
                         split into, two channels are connected only if they
                         belong to the same group.
        :type image_shape: tuple of 3 integers
        :type weights: 3D numpy.ndarray or theano.tensor
        :type filter_shape: tuple of 3 integers
        :type biases: 1D numpy.ndarray or theano.vector
        :type stride: pair of integers
        :type padding: pair of integers
        :type n_groups: integer
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_d_relu(self, activation):
        """Returns result of operation on given Numlike.

        :param Numlike activation: activation of relu layer
        :returns: Impact of input of relu on output of network
        :rtype: Numlike
        """
        raise NotImplementedError

    @staticmethod
    def derest_output(n_outputs):
        """Generates Numlike of impact of output on output.

        :param int n_outputs: Number of outputs of network.
        :returns: 2D square Numlike in shape (n_batches, n_outputs) with one
                  different "1" in every batch.
        :rtype: Numlike
        """
        raise NotImplementedError

