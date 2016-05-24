"""Intervals implemented in Numpy including special functions for
sparsifying.

This module contains NpInterval class and auxiliary objects.
"""
from theano import function
from theano import tensor as T

from itertools import product
import numpy as np
import math

from athenet.algorithm.numlike import Interval


class NpInterval(Interval):

    @staticmethod
    def construct(lower, upper):
        return NpInterval(lower, upper)


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

    def abs(self):
        """Returns absolute value of NpInterval.

        :rtype: NpInterval
        """
        lower = np.select([self.lower > 0.0, self.upper < 0.0, True],
                          [self.lower, -self.upper, 0.0])
        upper = np.maximum(-self.lower, self.upper)
        return NpInterval(lower, upper)

    @classmethod
    def from_shape(cls, shp, neutral=True, lower_val=None, upper_val=None):
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
            lower_val = cls.NEUTRAL_LOWER if neutral else cls.DEFAULT_LOWER
        if upper_val is None:
            upper_val = cls.NEUTRAL_UPPER if neutral else cls.DEFAULT_UPPER
        if lower_val > upper_val:
            if lower_val != np.inf or upper_val != -np.inf:
                raise ValueError("lower_val > upper_val")
        lower = np.full(shp, lower_val)
        upper = np.full(shp, upper_val)
        return NpInterval(lower, upper)

    @staticmethod
    def _reshape_for_padding(layer_input, image_shape, batch_size, padding,
                             value=0.0):

        h, w, n_channels = image_shape

        if padding == (0, 0):
            return np.broadcast_to(layer_input, (batch_size, n_channels, h, w))

        pad_h, pad_w = padding
        h_in = h + 2 * pad_h
        w_in = w + 2 * pad_w

        extra_pixels = np.full((batch_size, n_channels, h_in, w_in), value)
        extra_pixels[:, :, pad_h:(pad_h+h), pad_w:(pad_w+w)] = layer_input
        #maybe it will need broadcast_to too
        return extra_pixels

    def eval(self, *args):
        """Returns some readable form of stored value."""
        return self

    def op_relu(self):
        """Returns result of relu operation on given NpInterval.

        :rtype: NpInterval
        """
        lower = np.maximum(self.lower, 0.0)
        upper = np.maximum(self.upper, 0.0)
        return NpInterval(lower, upper)

    def op_softmax(self, input_shp):
        """Returns result of softmax operation on given NpInterval.

        :param integer input_shp: shape of 1D input
        :rtype: NpInterval
        """
        result = NpInterval.from_shape(input_shp, neutral=True)
        for i in xrange(input_shp):
            input_low = (self - self.upper[i]).exp()
            input_upp = (self - self.lower[i]).exp()
            sum_low = NpInterval.from_shape(1, neutral=True)
            sum_upp = NpInterval.from_shape(1, neutral=True)
            for j in xrange(input_shp):
                if j != i:
                    sum_low = sum_low + input_low[j]
                    sum_upp = sum_upp + input_upp[j]
            # Could consider evaluation below but it gives wrong answers.
            # It might be because of arithmetic accuracy.
            # sum_low = input_low.sum() - input_low[i]
            # sum_upp = input_upp.sum() - input_upp[i]
            upper_counter_low = input_low.upper[i]
            lower_counter_upp = input_upp.lower[i]
            upper_low = upper_counter_low / \
                (sum_low[0].lower + upper_counter_low)
            lower_upp = lower_counter_upp / \
                (sum_upp[0].upper + lower_counter_upp)
            result[i] = NpInterval(lower_upp, upper_low)
        return result

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
        :rtype: NpInterval
        """
        alpha /= local_range
        half = local_range / 2
        x = self
        sq = self.square()
        n_channels, h, w = input_shape
        extra_channels = self.from_shape((n_channels + 2 * half, h, w),
                                         neutral=True)
        extra_channels[half:half + n_channels, :, :] = sq
        s = self.from_shape(input_shape, neutral=True)

        for i in xrange(local_range):
            if i != half:
                s += extra_channels[i:i + n_channels, :, :]

        c = s * alpha + k

        def norm((arg_x, arg_c)):
            return arg_x / np.power(arg_c + alpha * np.sqr(arg_x), beta)

        def in_range((range_), val):
            return np.logical_and(np.less(range_.lower, val),
                                  np.less(val, range_.upper))

        def c_extr_from_x(arg_x):
            return np.square(arg_x) * ((2 * beta - 1) * alpha)

        def x_extr_from_c(arg_c):
            return np.sqrt(arg_c / ((2 * beta - 1) * alpha))

        res = NpInterval.from_shape(input_shape, lower_val=np.inf,
                                    upper_val=-np.inf)
        corners = [(x.lower, c.lower), (x.lower, c.upper),
                   (x.upper, c.lower), (x.upper, c.upper)]
        for corner in corners:
            res.lower = np.minimum(res.lower, norm(corner))
            res.upper = np.maximum(res.upper, norm(corner))
        maybe_extrema = [
            (0, c.lower), (0, c.upper),
            (x_extr_from_c(c.lower), c.lower),
            (x_extr_from_c(c.upper), c.upper),
            (x_extr_from_c(c.lower) * (-1), c.lower),
            (x_extr_from_c(c.upper) * (-1), c.upper),
            (x.lower, c_extr_from_x(x.lower)),
            (x.upper, c_extr_from_x(x.upper))
        ]
        extrema_conds = [
            in_range(x, maybe_extrema[0][0]),
            in_range(x, maybe_extrema[1][0]),
            in_range(x, maybe_extrema[2][0]),
            in_range(x, maybe_extrema[3][0]),
            in_range(x, maybe_extrema[4][0]),
            in_range(x, maybe_extrema[5][0]),
            in_range(c, maybe_extrema[6][1]),
            in_range(c, maybe_extrema[7][1])
        ]
        for m_extr, cond in zip(maybe_extrema, extrema_conds):
            norm_res = norm(m_extr)
            res.lower = np.select([cond, True],
                                  [np.minimum(res.lower, norm_res), res.lower])
            res.upper = np.select([cond, True],
                                  [np.maximum(res.upper, norm_res), res.upper])
        return res

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
        if True:
            l, u = T.tensor3(), T.tensor3()
            l2, u2 = self._theano_op_conv(l, u, weights, image_shape, filter_shape,
                                      biases, stride, padding, n_groups)
            self.op_conv_function = function(
                [l, u],
                [l2, u2]
            )

        lower, upper = self.op_conv_function(self.lower, self.upper)

        return NpInterval(lower, upper)

    def op_d_relu(self, activation):
        """Returns estimated impact of input of relu layer on output of
        network.

        :param NpInterval activation: estimated activation of input
        :param NpInterval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :returns: Estimated impact of input on output of network
        :rtype: NpInterval
        """
        lower_than_zero = activation.upper <= 0.
        contains_zero = np.logical_and(activation.lower < 0,
                                       activation.upper > 0)

        der_with_zero_l = np.minimum(self.lower, 0.)
        der_with_zero_u = np.maximum(self.upper, 0.)

        result_l = np.select([lower_than_zero, contains_zero, True],
                             [0., der_with_zero_l, self.lower])
        result_u = np.select([lower_than_zero, contains_zero, True],
                             [0., der_with_zero_u, self.upper])
        return NpInterval(result_l, result_u)

    @staticmethod
    def select(bool_list, interval_list):
        lower = [a.lower if isinstance(a, NpInterval) else a for a in interval_list]
        upper = [a.upper if isinstance(a, NpInterval) else a for a in interval_list]
        return NpInterval(np.select(bool_list, lower), np.select(bool_list, upper))

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
        # n_batches, n_in, h, w - number of batches, number of channels,
        # image height, image width
        n_batches, n_in, h, w = input_shape

        pad_h, pad_w = padding
        activation = activation.reshape_for_padding(input_shape, padding,
                                                    lower_val=-np.inf,
                                                    upper_val=-np.inf)
        input_shape = (n_batches, n_in, h + 2 * pad_h, w + 2 * pad_w)
        h += 2 * pad_h
        w += 2 * pad_w

        # fh, fw - pool height, pool width
        fh, fw = poolsize
        stride_h, stride_w = stride
        output = self
        result = activation.from_shape(input_shape, neutral=True)

        for at_h, at_w in product(xrange(0, h - fh + 1, stride_h),
                                  xrange(0, w - fw + 1, stride_w)):
            # at_out_h - height of output corresponding to pool at position
            # at_h
            at_out_h = at_h / stride_h
            # at_out_w - width of output corresponding to pool at position
            # at_w
            at_out_w = at_w / stride_w

            for at_f_h, at_f_w in product(xrange(at_h, at_h + fh),
                                          xrange(at_w, at_w + fw)):
                # maximum lower and upper value of neighbours
                neigh_max_low = -np.inf
                neigh_max_upp = -np.inf
                neigh_max_itv = NpInterval(neigh_max_low, neigh_max_upp)
                act_slice = activation[:, :, at_f_h, at_f_w]

                # setting maximum lower and upper of neighbours
                for at_f_h_neigh, at_f_w_neigh in \
                        product(xrange(at_h, at_h + fh),
                                xrange(at_w, at_w + fw)):

                    if (at_f_h_neigh, at_f_w_neigh) != (at_f_h, at_f_w):
                        neigh_slice = activation[:, :, at_f_h_neigh,
                                                 at_f_w_neigh]
                        neigh_max_itv = neigh_max_itv.max(neigh_slice)

                # must have impact on output
                must = act_slice.lower > neigh_max_itv.upper
                # cannot have impact on output
                cannot = act_slice.upper < neigh_max_itv.lower
                # or might have impact on output


                output_slice = output[:, :, at_out_h, at_out_w]
                output_with_0 = NpInterval(np.minimum(output_slice.lower, 0.),
                                           np.maximum(output_slice.upper, 0.))

                result[:, :, at_f_h, at_f_w] += \
                    NpInterval.select([must, cannot, True],
                                      [output_slice, 0., output_with_0])

        return result[:, :, pad_h:h - pad_h, pad_w:w - pad_w]

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
        # n_batches, n_in, h, w - number of batches, number of channels,
        # image height, image width
        n_batches, n_in, h, w = input_shape

        pad_h, pad_w = padding
        activation = activation.reshape_for_padding(input_shape, padding,
                                                    lower_val=0.,
                                                    upper_val=0.)
        input_shape = (n_batches, n_in, h + 2 * pad_h, w + 2 * pad_w)
        h += 2 * pad_h
        w += 2 * pad_w

        # fh, fw - pool height, pool width
        fh, fw = poolsize
        stride_h, stride_w = stride
        output = self
        result = activation.from_shape(input_shape, neutral=True)

        for at_h, at_w in product(xrange(0, h - fh + 1, stride_h),
                                  xrange(0, w - fw + 1, stride_w)):
            # at_out_h - height of output corresponding to pool at position
            # at_h
            at_out_h = at_h / stride_h
            # at_out_w - width of output corresponding to pool at position
            # at_w
            at_out_w = at_w / stride_w

            output_slice = output[:, :, at_out_h, at_out_w].\
                reshape((n_batches, n_in, 1, 1))

            result_slice = result[:, :, at_h:at_h + fh, at_w:at_w + fw]
            result_slice += output_slice
            result[:, :, at_h:at_h + fh, at_w:at_w + fw] = result_slice

        result /= np.prod(poolsize)
        return result[:, :, pad_h:h - pad_h, pad_w:w - pad_w]

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
                (-math.sqrt(3 * c) / math.sqrt(alpha * (2 * beta - 1)), c)
                for c in [c_low, c_up]]
            possibilities_c2 = [
                (math.sqrt(3 * c) / math.sqrt(alpha * (2 * beta - 1)), c)
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
            sqrt1 = [(math.sqrt((c + a * y ** 2) / (a * (2 * b + 1))), y, c)
                     for y, c in product([y_low, y_up], [c_low, c_up])]
            sqrt2 = [(-math.sqrt((c + a * y ** 2) / (a * (2 * b + 1))), y, c)
                     for y, c in product([y_low, y_up], [c_low, c_up])]
            return [(x, y, c) for x, y, c in sqrt1 + sqrt2
                    if x_low <= x <= x_up]

        def extremas_3d_dy(x_low, x_up, y_low, y_up, c_low, c_up):
            # ddf/dx/dy = 0
            # a*x**2=a(2*b+1)*y**2-c
            a = alpha
            b = beta
            sqrt1 = [(x, math.sqrt((c + a * x ** 2) / (a * (2 * b + 1))), c)
                     for x, c in product([x_low, x_up], [c_low, c_up])]
            sqrt2 = [(x, -math.sqrt((c + a * x ** 2) / (a * (2 * b + 1))), c)
                     for x, c in product([x_low, x_up], [c_low, c_up])]
            return [(x, y, c) for x, y, c in sqrt1 + sqrt2
                    if y_low <= y <= y_up]

        def extremas_3d_dxdy(x_low, x_up, y_low, y_up, c_low, c_up):
            # ddf/dx/dy = 0 && ddf/dx/dx = 0
            vals_cl = [sign * math.sqrt(c_low / (2 * alpha * beta))
                       for sign in [-1, 1]]
            vals_cu = [sign * math.sqrt(c_up / (2 * alpha * beta))
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
            der = NpInterval(np.inf, -np.inf)
            for x, c in extremas:
                val = der_eq(x, c)
                if der.lower > val:
                    der.lower = val
                if der.upper < val:
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

                    der = NpInterval(np.inf, -np.inf)
                    for x, y, c in extremas:
                        val = der_not_eq(x, y, c)
                        if der.lower > val:
                            der.lower = val
                        if der.upper < val:
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

        # n_in, h, w - number of input channels, image height, image width
        n_batches, n_in, h, w = input_shape
        # n_out, fh, fw - number of output channels, filter height, filter
        # width
        n_out, fh, fw = filter_shape
        pad_h, pad_w = padding
        output = self

        # g_in - number of input channels per group
        g_in = n_in / n_groups
        # g_out - number of output channels per group
        g_out = n_out / n_groups
        stride_h, stride_w = stride
        h += 2 * pad_h
        w += 2 * pad_w
        padded_input_shape = (n_batches, n_in, h, w)
        result = NpInterval.from_shape(padded_input_shape, neutral=True)

        # see: flipping kernel
        # in theano.conv_2d flipped kernel is used
        weights = weights[:, :, ::-1, ::-1]
        weights_neg = np.minimum(weights, 0.0)
        weights_pos = np.maximum(weights, 0.0)

        for at_g in xrange(n_groups):
            # beginning and end of at_g'th group of input channel in input
            at_in_from = at_g * g_in
            at_in_to = at_in_from + g_in
            # beginning and end of at_g'th group of output channel in weights
            # note: amount of input and output group are equal
            at_out_from = at_g * g_out
            at_out_to = at_out_from + g_out

            for at_h, at_w in product(xrange(0, h - fh + 1, stride_h),
                                      xrange(0, w - fw + 1, stride_w)):
                # at_out_h - height of output corresponding to filter at
                # position at_h
                at_out_h = at_h / stride_h
                # at_out_w - height of output corresponding to filter at
                # position at_w
                at_out_w = at_w / stride_w

                # weights slice that impacts on (at_out_h, at_out_w) in
                # output
                weights_pos_slice = weights_pos[at_out_from:at_out_to, :, :, :]
                weights_neg_slice = weights_neg[at_out_from:at_out_to, :, :, :]
                # shape of weights_slice: (g_out, g_in, h, w)

                # slice of output
                out_slice_low = output.lower[:, at_out_from:at_out_to,
                                             at_out_h, at_out_w]
                out_slice_low = \
                    out_slice_low.reshape((n_batches, g_out, 1, 1, 1))
                out_slice_upp = output.upper[:, at_out_from:at_out_to,
                                             at_out_h, at_out_w]
                out_slice_upp = \
                    out_slice_upp.reshape((n_batches, g_out, 1, 1, 1))

                # results
                res_low_pos = (out_slice_low * weights_pos_slice).sum(axis=1)
                res_low_neg = (out_slice_low * weights_neg_slice).sum(axis=1)
                res_upp_pos = (out_slice_upp * weights_pos_slice).sum(axis=1)
                res_upp_neg = (out_slice_upp * weights_neg_slice).sum(axis=1)

                res_slice_lower = res_low_pos + res_upp_neg
                res_slice_upper = res_upp_pos + res_low_neg
                res_slice = NpInterval(res_slice_lower, res_slice_upper)

                result[:, at_in_from:at_in_to, at_h:(at_h + fh),
                       at_w:(at_w + fw)] += res_slice

        # remove padding
        result = result[:, :, pad_h:(h - pad_h), pad_w:(w - pad_w)]
        return result

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
