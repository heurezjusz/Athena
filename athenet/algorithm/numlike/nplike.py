"""numpy.ndarray packed in Numlike interface."""

from numlike import Numlike
import numpy


class Nplike(Numlike):

    def __init__(self, value):
        super(Nplike, self).__init__()
        self.value = value

    def __getitem__(self, at):
        return Nplike(self.value[at])

    def __setitem__(self, at, other):
        self.value[at] = other.value

    @property
    def shape(self):
        return self.value.shape

    def __add__(self, other):
        if isinstance(other, Nplike):
            return Nplike(self.value + other.value)
        else:
            return Nplike(self.value + other)

    def __sub__(self, other):
        if isinstance(other, Nplike):
            return Nplike(self.value - other.value)
        else:
            return Nplike(self.value - other)

    def __mul__(self, other):
        if isinstance(other, Nplike):
            return Nplike(self.value * other.value)
        else:
            return Nplike(self.value * other)

    def __div__(self, other):
        if isinstance(other, Nplike):
            return Nplike(self.value / other.value)
        else:
            return Nplike(self.value / other)

    def __rdiv__(self, other):
        if isinstance(other, Nplike):
            return Nplike(other.value / self.value)
        else:
            return Nplike(other / self.value)

    def reciprocal(self):
        return Nplike(numpy.reciprocal(self.value))

    def neg(self):
        return Nplike(-self.value)

    def exp(self):
        return Nplike(numpy.exp(self.value))

    def square(self):
        return Nplike(numpy.square(self.value))

    def power(self, exponent):
        return Nplike(numpy.power(self.value, exponent))

    def dot(self, other):
        """Dot product of self and other.

        :param numpy.ndarray other: second argument of product
        """
        return Nplike(numpy.dot(self.value, other))

    def max(self, other):
        if isinstance(other, Nplike):
            return Nplike(numpy.maximum(self.value, other.value))
        else:
            return Nplike(numpy.maximum(self.value, other))

    def amax(self, axis=None, keepdims=False):
        return Nplike(numpy.amax(self.value, axis=axis, keepdims=keepdims))

    def reshape(self, shape):
        return Nplike(self.value.reshape(shape))

    def reshape_for_padding(self, *args):
        return super(Nplike, self).reshape_for_padding(*args)

    def flatten(self):
        return Nplike(self.value.flatten())

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Sum of array elements over a given axis like in numpy.ndarray.

        :param integer or None axis: axis along which this function sums
        :param type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        """
        s = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        if isinstance(s, numpy.ndarray):
            return Nplike(s)
        else:
            return Nplike(numpy.array([s]))

    def abs(self):
        return Nplike(numpy.absolute(self.value))

    @property
    def T(self):
        return Nplike(self.value.T)

    @staticmethod
    def from_shape(shp, neutral=True):
        if neutral:
            return Nplike(numpy.zeros(shp))
        else:
            return Nplike(numpy.ones(shp))

    def eval(self):
        return self.value

    @staticmethod
    def derest_output(self, *args):
        return super(Nplike, self).derest_output(*args)

    def __repr__(self):
        """Standard repr method."""
        return repr(self.value)

    def __str__(self):
        """"Standard str method."""
        return str(self.value)

    def concat(self, other, axis=0):
        return Nplike(numpy.concatenate((self.value, other.value), axis=axis))
