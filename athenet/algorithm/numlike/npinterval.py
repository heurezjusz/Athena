"""Intervals implemented in Numpy including special functions for
sparsifying.

This module contains NpInterval class and auxiliary objects.
"""

from athenet.algorithm.numlike import Numlike
from itertools import product
import numpy as np

NEUTRAL_INTERVAL_LOWER = 0.0
NEUTRAL_INTERVAL_UPPER = 0.0
NEUTRAL_INTERVAL_VALUES = (NEUTRAL_INTERVAL_LOWER, NEUTRAL_INTERVAL_UPPER)

DEFAULT_INTERVAL_LOWER = 0.0
DEFAULT_INTERVAL_UPPER = 255.0
DEFAULT_INTERVAL_VALUES = (DEFAULT_INTERVAL_LOWER, DEFAULT_INTERVAL_UPPER)


class NpInterval(Numlike):
    def __init__(self, lower=None, upper=None):
        """
        :param numpy.ndarray lower:
        :param numpy.ndarray upper:
        :return:
        """
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        """Returns specified slice of NpInterval.

        :at: Coordinates / slice to be taken.
        :rtype: NpInterval
        """
        return NpInterval(self.lower[at], self.upper[at])

    def __setitem__(self, at, other):
        """Just like numpy __setitem__ function, but as a operator.
        :at: Coordinates / slice to be set.
        :other: Data to be put at 'at'.
        """
        self.lower[at] = other.lower
        self.upper[at] = other.upper

    def __str__(self):
        return "vvvvv\n" + self.lower.__str__() + "\n=====\n" + \
               self.upper.__str__() + "\n^^^^^"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        """Returns shape of numlike.

        :rtype: tuple of integers
        """
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two NpIntervals.

        :param other: value to be added.
        :type other: NpInterval or numpy.array or float
        :rtype: NpInterval
        """
        if isinstance(other, NpInterval):
            res_lower = self.lower + other.lower
            res_upper = self.upper + other.upper
        else:
            res_lower = self.lower + other
            res_upper = self.upper + other
        return NpInterval(res_lower, res_upper)

    def antiadd(self, other):
        """For given NpInterval returns NpInterval which shuold be added
        to id to get NpInterval equal to self.

        :param other: NpInterval which was added.
        :type other: NpInterval
        :rtype: NpInterval
        """
        return NpInterval(self.lower - other.lower, self.upper - other.upper)

    def __sub__(self, other):
        """Returns difference between two numlikes.

        :param other: value to be subtracted.
        :type other: NpInterval or np.ndarray or float
        :rtype: NpInterval
        """
        if isinstance(other, NpInterval):
            res_lower = self.lower - other.upper
            res_upper = self.upper - other.lower
        else:
            res_lower = self.lower - other
            res_upper = self.upper - other
        return NpInterval(res_lower, res_upper)

    def __mul__(self, other):
        """Returns product of two NpIntervals

        :param other: value to be multiplied.
        :type other: NpInterval or numpy.array or float
        :rtype: NpInterval
        """
        if isinstance(other, NpInterval):
            ll = self.lower * other.lower
            lu = self.lower * other.upper
            ul = self.upper * other.lower
            uu = self.upper * other.upper
            lower = np.minimum(np.minimum(ll, lu), np.minimum(ul, uu))
            upper = np.maximum(np.maximum(ll, lu), np.maximum(ul, uu))
        else:
            ll = self.lower * other
            uu = self.upper * other
            lower = np.minimum(ll, uu)
            upper = np.maximum(ll, uu)
        return NpInterval(lower, upper)

    def __div__(self, other):
        """Returns quotient of self and other.

        :param other: divisor
        :type other: NpInterval or numpy.ndarray or float
        :rtype: NpInterval

        .. warning:: Divisor should not contain zero.
        """
        if isinstance(other, NpInterval):
            ll = self.lower / other.lower
            lu = self.lower / other.upper
            ul = self.upper / other.lower
            uu = self.upper / other.upper
            lower = np.minimum(np.minimum(ll, lu), np.minimum(ul, uu))
            upper = np.maximum(np.maximum(ll, lu), np.maximum(ul, uu))
            return NpInterval(lower, upper)
        else:
            lower = self.lower / other
            upper = self.upper / other
            return NpInterval(np.minimum(lower, upper),
                              np.maximum(lower, upper))

    def __rdiv__(self, other):
        """Returns quotient of other and self.

        :param other: dividend
        :type other: float
        :rtype: Nplike
        .. warning:: divisor (self) should not contain zero, other must be
                     float
        """
        if isinstance(other, NpInterval):
            # Should never happen. __div__ should be used instead.
            raise NotImplementedError
        else:
            if other > 0:
                return NpInterval(other / self.upper, other / self.lower)
            else:
                return NpInterval(other / self.lower, other / self.upper)

    def reciprocal(self):
        """Returns reciprocal (1/x) of the NpInterval.

        :rtype: NpInterval
        """
        upper_reciprocal = np.reciprocal(self.upper)
        lower_reciprocal = np.reciprocal(self.lower)
        return NpInterval(np.minimum(upper_reciprocal, lower_reciprocal),
                          np.maximum(upper_reciprocal, lower_reciprocal))

    def neg(self):
        """Returns (-1) * NpInterval

        :rtype: NpInterval
        """
        return NpInterval(np.negative(self.upper), np.negative(self.lower))

    def exp(self):
        """Returns NpInterval representing the exponential of the Numlike.

        :rtype: NpInterval
        """
        return NpInterval(np.exp(self.lower), np.exp(self.upper))

    def _has_zero(self):
        """For any interval in NpInterval, returns whether is contains zero.

        :rtype: numpy.array of Boolean
        """
        return np.logical_and(self.lower <= 0, self.upper >= 0)

    def square(self):
        """Returns square of the NpInterval

        :rtype: NpInterval
        """
        uu = self.upper * self.upper
        ll = self.lower * self.lower
        lower = np.select([self._has_zero(), True], [0, np.minimum(ll, uu)])
        upper = np.maximum(ll, uu)
        return NpInterval(lower, upper)

    def power(self, exponent):
        """Returns NpInterval^exponent.

        :param float exponent: Number to be passed as exponent to N^exponent.
        :rtype: NpInterval
        """
        le = np.power(self.lower, exponent)
        ue = np.power(self.upper, exponent)
        if isinstance(exponent, (int, long)):
            if exponent > 0:
                if exponent % 2 == 0:
                    l = np.select([self._has_zero(), True],
                                  [0, np.minimum(le, ue)])
                    u = np.maximum(le, ue)
                else:
                    l = le
                    u = ue
            else:
                if exponent % 2 == 0:
                    l = np.minimum(le, ue)
                    u = np.maximum(le, ue)
                else:
                    l = ue
                    u = le
        else:
            # Assumes self.lower >= 0. Otherwise it is incorrectly defined.
            # There is no check.
            if exponent > 0:
                l = le
                u = ue
            else:
                l = ue
                u = le
        return NpInterval(l, u)

    def dot(self, other):
        """Dot product of NpInterval and a other.

        :param numpy.ndarray other: second dot param
        :rtype: NpInterval
        """
        other_negative = np.minimum(other, 0.0)
        other_positive = np.maximum(other, 0.0)
        lower_pos_dot = np.dot(self.lower, other_positive)
        lower_neg_dot = np.dot(self.lower, other_negative)
        upper_pos_dot = np.dot(self.upper, other_positive)
        upper_neg_dot = np.dot(self.upper, other_negative)
        return NpInterval(lower_pos_dot + upper_neg_dot,
                          upper_pos_dot + lower_neg_dot)

    def max(self, other):
        """Returns interval such that for any numbers (x, y) in a pair of
        corresponding intervals in (self, other) arrays, max(x, y) is in result
        and no other.

        :param other: interval to be compared
        :type other: NpInterval or numpy.ndarray
        :rtype: NpInterval
        """
        if isinstance(other, NpInterval):
            return NpInterval(np.maximum(self.lower, other.lower),
                              np.maximum(self.upper, other.upper))
        else:
            return NpInterval(np.maximum(self.lower, other),
                              np.maximum(self.upper, other))

    def amax(self, axis=None, keepdims=False):
        """Returns maximum of a NpInterval along an axis.

        :param axis: axis along which max is evaluated
        :param Boolean keepdims: whether flattened dimensions should remain
        :rtype: NpInterval
        """
        lower = self.lower.max(axis=axis, keepdims=keepdims)
        upper = self.upper.max(axis=axis, keepdims=keepdims)
        return NpInterval(lower, upper)

    def reshape(self, shape):
        """Reshapes NpInterval

        :param tuple shape:
        """
        return NpInterval(self.lower.reshape(shape),
                          self.upper.reshape(shape))

    def flatten(self):
        """Flattens NpInterval

        :rtype: NpInterval
        """
        return NpInterval(self.lower.flatten(),
                          self.upper.flatten())

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Sum of array elements over a given axis like in numpy.ndarray.

        :param axis: axis along which this function sums
        :param numeric type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        :type axis: integer, tuple of integers or None
        :rtype: NpInterval

        """
        return NpInterval(
            self.lower.sum(axis=axis, dtype=dtype, keepdims=keepdims),
            self.upper.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        )

    def abs(self):
        """Returns absolute value of NpInterval.

        :rtype: NpInterval
        """
        lower = np.select([self.lower > 0.0, self.upper < 0.0, True],
                          [self.lower, -self.upper, 0.0])
        upper = np.maximum(-self.lower, self.upper)
        return NpInterval(lower, upper)

    @property
    def T(self):
        """Tensor transposition like in numpy.ndarray.

        :rtype: NpInterval
        """
        return NpInterval(self.lower.T, self.upper.T)

    @staticmethod
    def from_shape(shp, neutral=True, lower_val=None, upper_val=None):
        """Returns NpInterval of shape shp with given lower and upper values.

        :param tuple of integers or integer shp : shape of created NpInterval
        :param Boolean neutral: if True sets (lower_val, upper_val) to
                                NEUTRAL_INTERVAL_VALUES, otherwise to
                                DEFAULT_INTERVAL_VALUES, works only if pair is
                                not set by passing arguments.
        :param float lower_val: value of lower bound
        :param float upper_val: value of upper bound
        """
        if lower_val is None:
            lower_val = NEUTRAL_INTERVAL_LOWER if neutral else \
                DEFAULT_INTERVAL_LOWER
        if upper_val is None:
            upper_val = NEUTRAL_INTERVAL_UPPER if neutral else \
                DEFAULT_INTERVAL_UPPER
        if lower_val > upper_val:
            if lower_val != np.inf or upper_val != -np.inf:
                raise ValueError("lower_val > upper_val")
        lower = np.full(shp, lower_val)
        upper = np.full(shp, upper_val)
        return NpInterval(lower, upper)

    @staticmethod
    def _reshape_for_padding(layer_input, image_shape, batch_size, padding,
                             value=0.0):
        if padding == (0, 0):
            return layer_input

        h, w, n_channels = image_shape
        pad_h, pad_w = padding
        h_in = h + 2 * pad_h
        w_in = w + 2 * pad_w

        extra_pixels = np.full((batch_size, n_channels, h_in, w_in), value)
        extra_pixels[:, :, pad_h:(pad_h+h), pad_w:(pad_w+w)] = layer_input
        return extra_pixels

    def reshape_for_padding(self, shape, padding, lower_val=None,
                            upper_val=None):
        """Returns padded Numlike.

        :param tuple of 4 integers shape: shape of input in format
                                          (batch size, number of channels,
                                           height, width)
        :param pair of integers padding: padding to be applied
        :param float lower_val: value of lower bound in new fields
        :param float upper_val: value of upper bound in new fields
        :returns: padded layer_input
        :rtype: NpInterval
        """
        if lower_val is None:
            lower_val = NEUTRAL_INTERVAL_LOWER
        if upper_val is None:
            upper_val = NEUTRAL_INTERVAL_UPPER
        n_batches, n_in, h, w = shape
        padded_low = self._reshape_for_padding(self.lower, (h, w, n_in),
                                               n_batches, padding, lower_val)
        padded_upp = self._reshape_for_padding(self.upper, (h, w, n_in),
                                               n_batches, padding, upper_val)
        return NpInterval(padded_low, padded_upp)

    def eval(self, *args):
        """Returns some readable form of stored value."""
        return self

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
        """Returns estimated impact of input of relu layer on output of
        network.

        :param Numlike activation: estimated activation of input
        :param Numlike self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :returns: Estimated impact of input on output of network
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_d_max_pool(self, activation, input_shape, poolsize, stride,
                      padding):
        """Returns estimated impact of max pool layer on output of network.

        :param Numlike self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param Numlike activation: estimated activation of input
        :param input_shape: shape of layer input in format (batch size,
                            number of channels, height, width)
        :type input_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width),
                                          not equal (1, 1)
        :param pair of integers stride: stride of max pool
        :param pair of integers padding: padding of max pool
        :returns: Estimated impact of input on output of network
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_d_avg_pool(self, activation, input_shape, poolsize, stride,
                      padding):
        """Returns estimated impact of avg pool layer on output of network.

        :param Numlike self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param Numlike activation: estimated activation of input
        :param input_shape: shape of layer input in format (batch size,
                            number of channels, height, width)
        :type input_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width),
                                          not equal (1, 1)
        :param pair of integers stride: stride of avg pool
        :param pair of integers padding: padding of avg pool
        :returns: Estimated impact of input on output of network
        :rtype: Numlike
        """
        raise NotImplementedError

    def op_d_norm(self, activation, input_shape, local_range, k, alpha,
                  beta):
        """Returns estimated impact of input of norm layer on output of
        network.

        :param NpInterval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param NpInterval activation: estimated activation of input
        :param input_shape: shape of layer input in format (batch size,
                            number of channels, height, width)
        :type input_shape: tuple of 4 integers
        :param integer local_range: size of local range in local range
                                    normalization
        :param float k: local range normalization k argument
        :param float alpha: local range normalization alpha argument
        :param float beta: local range normalization beta argument
        :rtype: NpInterval
        """
        result = NpInterval(np.zeros(input_shape),
                            np.zeros(input_shape))
        activation_sqares = activation.square()
        local_range /= 2


        # some piece of math, unnecessary in any other place:
        # derivative for x placed in denominator of norm function
        def der_eq(x, c):
            """
            Return derivative of norm function for value in denominator
            :param x: value in denominator
            :param c: k + sum of squares of other values

            In this representation norm function equals to
            x / (c + alpha * (x ** 2)) ** beta

            :return: value of derivative of norm function
            """
            return (alpha * (1 - 2 * beta) * x ** 2 + c) / \
                   (alpha * x ** 2 + c) ** (beta + 1)


        # possible extremas
        def root1_2d(c_low, c_up, x_low, x_up):
            # df / dx = 0
            # returns roots of derivative of derivetive of norm function
            # x = 0
            # intersects solution rectangle with x = 0

            possibilities_c0 = [(0., c) for c in [c_low, c_up]]
            possibilities_c1 = [
                (-(3 * c)**0.5 / (alpha * (2 * beta - 1))**0.5, c)
                for c in [c_low, c_up]]
            possibilities_c2 = [
                ((3 * c)**0.5 / (alpha * (2 * beta - 1))**0.5, c)
                for c in [c_low, c_up]]

            return [(x, c) for x, c in possibilities_c0 + possibilities_c1
                    + possibilities_c2 if x_low <= x <= x_up]


        def root2_2d(c_low, c_up, x_low, x_up):
            # df / dc = 0
            # returns roots of derivative of derivetive of norm function
            # x = - sqrt(c) / sqrt (alpha * (2*beta+1))
            # intersects solution rectangle with parabola above

            possibilities_x = [(x, alpha * (2 * beta + 1) * x ** 2)
                               for x in [x_low, x_up]]

            return [(x, c) for x, c in possibilities_x
                    if c_low <= c and c <= c_up]

        # derivative for x not from denominator
        def der_not_eq(x, y, c):
            """
            Returns value of derivative of norm function for element not
            placed in derivative
            :param x: element to compute derivative after
            :param y: element placed in denominator
            :param c: k + alpha * sum of squares of other elements

            In this representation norm function equals to
            y / (c + aplha * x**2 + alpha * y**2) ** beta

            :return: Returns value of derivative of norm function
            """
            return -2 * alpha * beta * x * y / \
                   (c + alpha * (x ** 2 + y ** 2)) ** (beta + 1)


        # possible extremas of this derivative
        def extremas_3d(x_low, x_up, y_low, y_up, c_low, c_up):
            return [(x, y, c) for x, y, c in
                    product([x_low, x_up], [y_low, y_up], [c_low, c_up])
                    if x_low <= x <= x_up and y_low <= y <= y_up]


        def extremas_3d_dx(x_low, x_up, y_low, y_up, c_low, c_up):
            # ddf/dx/dx = 0
            # a*y**2=a(2*b+1)*x**2-c
            a = alpha
            b = beta
            sqrt1 = [(((c + a * y ** 2) / (a * (2 * b + 1)))**0.5, y, c)
                     for y, c in product([y_low, y_up], [c_low, c_up])]
            sqrt2 = [(-((c + a * y ** 2) / (a * (2 * b + 1)))**0.5, y, c)
                     for y, c in product([y_low, y_up], [c_low, c_up])]
            return [(x, y, c) for x, y, c in sqrt1 + sqrt2
                    if x_low <= x <= x_up]


        def extremas_3d_dy(x_low, x_up, y_low, y_up, c_low, c_up):
            # ddf/dx/dy = 0
            # a*x**2=a(2*b+1)*y**2-c
            a = alpha
            b = beta
            sqrt1 = [(x, ((c + a * x ** 2) / (a * (2 * b + 1)))**0.5, c)
                     for x, c in product([x_low, x_up], [c_low, c_up])]
            sqrt2 = [(x, -((c + a * x ** 2) / (a * (2 * b + 1)))**0.5, c)
                     for x, c in product([x_low, x_up], [c_low, c_up])]
            return [(x, y, c) for x, y, c in sqrt1 + sqrt2
                    if y_low <= y <= y_up]


        def extremas_3d_dxdy(x_low, x_up, y_low, y_up, c_low, c_up):
            # ddf/dx/dy = 0 && ddf/dx/dx = 0
            vals_cl = [sign * (c_low / (2 * alpha * beta))**0.5
                       for sign in [-1, 1]]
            vals_cu = [sign * (c_up / (2 * alpha * beta))**0.5
                       for sign in [-1, 1]]

            pts_low = [(x, y, c_low) for x, y in product(vals_cl, vals_cl)]
            pts_up = [(x, y, c_up) for x, y in product(vals_cu, vals_cu)]

            return [(x, y, c) for x, y, c in pts_low + pts_up
                    if x_low <= x <= x_up and y_low <= y <= y_up]

        batches, channels, h, w = input_shape
        for b, channel, at_h, at_w in product(xrange(batches), xrange(channels),
                                              xrange(h), xrange(w)):
            C = NpInterval(np.asarray([k]), np.asarray([k]))
            for i in xrange(-local_range, local_range + 1):
                if channels > i + channel >= 0 != i:
                    C += activation_sqares[b][channel + i][at_h][at_w]

            Y = activation[b][channel][at_h][at_w]

            # eq case
            extremas = [(x, c) for x, c in product([Y.lower, Y.upper],
                                                   [C.lower, C.upper])]
            extremas.extend(root1_2d(C.lower, C.upper, Y.lower, Y.upper))
            extremas.extend(root2_2d(C.lower, C.upper, Y.lower, Y.upper))
            der = NpInterval()
            for x, c in extremas:
                val = der_eq(x, c)
                if der.lower is None or der.lower > val:
                    der.lower = val
                if der.upper is None or der.upper < val:
                    der.upper = val
            result[b][channel][at_h][at_w] += der * self[b][channel][at_h][at_w]

            # not_eq_case
            for i in xrange(-local_range, local_range + 1):
                if i != 0 and 0 <= (i + channel) < channels:
                    X = activation[b][channel + i][at_h][at_w]
                    X2 = activation_sqares[b][channel + i][at_h][at_w]
                    C = C.antiadd(X2)

                    extremas = extremas_3d(X.lower, X.upper, Y.lower,
                                           Y.upper, C.lower, C.upper) + \
                               extremas_3d_dx(X.lower, X.upper, Y.lower,
                                              Y.upper, C.lower, C.upper) + \
                               extremas_3d_dy(X.lower, X.upper, Y.lower,
                                              Y.upper, C.lower, C.upper) + \
                               extremas_3d_dxdy(X.lower, X.upper, Y.lower,
                                                Y.upper, C.lower, C.upper)

                    der = NpInterval()
                    for x, y, c in extremas:
                        val = der_not_eq(x, y, c)
                        if der.lower is None or der.lower > val:
                            der.lower = val
                        if der.upper is None or der.upper < val:
                            der.upper = val
                    result[b][channel + i][at_h][at_w] += \
                        der * self[b][channel][at_h][at_w]
                    C += X2

        return result

    def op_d_conv(self, input_shape, filter_shape, weights,
                  stride, padding, n_groups):
        """Returns estimated impact of input of convolutional layer on output
        of network.

        :param Numlike self: estimated impact of output of layer on output
                             of network in shape (batch_size,
                             number of channels, height, width)
        :param input_shape: shape of layer input in the format
                            (number of batches,
                             number of input channels,
                             image height,
                             image width)
        :type input_shape: tuple of 4 integers
        :param filter_shape: filter shape in the format
                             (number of output channels, filter height,
                              filter width)
        :type filter_shape: tuple of 3 integers
        :param weights: Weights tensor in format (number of output channels,
                                                  number of input channels,
                                                  filter height,
                                                  filter width)
        :type weights: numpy.ndarray or theano tensor
        :param stride: pair representing interval at which to apply the filters
        :type stride: pair of integers
        :param padding: pair representing number of zero-valued pixels to add
                        on each side of the input.
        :type padding: pair of integers
        :param n_groups: number of groups input and output channels will be
                         split into, two channels are connected only if they
                         belong to the same group.
        :type n_groups: integer
        :returns: Estimated impact of input on output of network
        :rtype: Numlike
        """
        raise NotImplementedError

    @staticmethod
    def derest_output(n_outputs):
        """Generates NpInterval of impact of output on output.

        :param int n_outputs: Number of outputs of network.
        :returns: 2D square NpInterval in shape (n_batches, n_outputs) with one
                  different "1" in every batch, like numpy.eye(n_outputs)
        :rtype: NpInterval
        """
        np_matrix = np.eye(n_outputs)
        return NpInterval(np_matrix, np_matrix)
