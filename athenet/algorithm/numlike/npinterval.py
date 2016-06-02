"""Intervals implemented in Numpy including special functions for
sparsifying.

This module contains NpInterval class and auxiliary objects.
"""
from theano import function, config
from theano import tensor as T

from itertools import product
import numpy as np
import math

from athenet.algorithm.numlike import Interval


class NpInterval(Interval):

    def __init__(self, lower, upper, accuracy=1e-6):
        """Creates NpInterval.

        :param numpy.ndarray lower: lower bound of Interval to be set
        :param numpy.ndarray upper: upper bound of Interval to be set
        :param float accuracy: acceptable error in check lower <= upper

        """
        assert (lower - accuracy <= upper).all()
        super(NpInterval, self).__init__(lower, upper)

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

    def _antiadd(self, other):
        """For given NpInterval returns NpInterval which shuold be added
        to id to get NpInterval equal to self.

        :param other: NpInterval which was added.
        :type other: NpInterval
        :rtype: NpInterval
        """
        return NpInterval(self.lower - other.lower, self.upper - other.upper)

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
        """Returns NpInterval representing the exponential of the NpInterval.

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
        lower = np.full(shp, lower_val, dtype=config.floatX)
        upper = np.full(shp, upper_val, dtype=config.floatX)
        return NpInterval(lower, upper)

    def broadcast(self, shape):
        """Broadcast interval

        :param shape: tuple of integers
        :rtype: NpInterval
        """
        return NpInterval(np.broadcast_to(self.lower, shape),
                          np.broadcast_to(self.upper, shape))

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

    @staticmethod
    def select(bool_list, interval_list):
        lower = [a.lower if isinstance(a, NpInterval) else a
                 for a in interval_list]
        upper = [a.upper if isinstance(a, NpInterval) else a
                 for a in interval_list]
        return NpInterval(np.select(bool_list, lower),
                          np.select(bool_list, upper))

    def concat(self, other, axis=0):
        lower = np.concatenate([self.lower, other.lower], axis=axis)
        upper = np.concatenate([self.upper, other.upper], axis=axis)
        return NpInterval(lower, upper)

    @staticmethod
    def stack(intervals, axis=0):
        lower = np.stack([i.lower for i in intervals], axis=axis)
        upper = np.stack([i.upper for i in intervals], axis=axis)
        return NpInterval(lower, upper)

    def eval(self, *args):
        """Returns some readable form of stored value."""
        return self.lower, self.upper

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
        result = NpInterval.from_shape((input_shp, ), neutral=True)
        for i in xrange(input_shp):
            input_low = (self - self.upper[i]).exp()
            input_upp = (self - self.lower[i]).exp()
            sum_low = NpInterval.from_shape((1, ), neutral=True)
            sum_upp = NpInterval.from_shape((1, ), neutral=True)
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
            return arg_x / np.power(arg_c + alpha * np.square(arg_x), beta)

        def in_range((range_), val):
            return np.logical_and(np.less(range_.lower, val),
                                  np.less(val, range_.upper))

        def c_extr_from_x(arg_x):
            return np.square(arg_x) * ((2 * beta - 1) * alpha)

        def x_extr_from_c(arg_c):
            return np.sqrt(arg_c / ((2 * beta - 1) * alpha))

        corner_lower = np.full(input_shape, np.inf)
        corner_upper = np.full(input_shape, -np.inf)
        corners = [(x.lower, c.lower), (x.lower, c.upper),
                   (x.upper, c.lower), (x.upper, c.upper)]
        for corner in corners:
            corner_lower = np.minimum(corner_lower, norm(corner))
            corner_upper = np.maximum(corner_upper, norm(corner))
        res = NpInterval(corner_lower, corner_upper)

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
        """Returns estimated activation of convolution applied to NpInterval.

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
        :rtype: NpInterval
        """

        t_lower, t_upper = T.tensor3(), T.tensor3()
        result_lower, result_upper = self._theano_op_conv(
            t_lower, t_upper, weights, image_shape, filter_shape,
            biases, stride, padding, n_groups
        )
        op_conv_function = function([t_lower, t_upper],
                                    [result_lower, result_upper])

        lower, upper = op_conv_function(
            self.lower,
            self.upper
        )
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
        lower = [a.lower if isinstance(a, NpInterval) else a
                 for a in interval_list]
        upper = [a.upper if isinstance(a, NpInterval) else a
                 for a in interval_list]
        return NpInterval(np.select(bool_list, lower),
                          np.select(bool_list, upper))

    def op_d_max_pool(self, activation, input_shape, poolsize, stride,
                      padding):
        """Returns estimated impact of max pool layer on output of network.

        :param NpInterval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param NpInterval activation: estimated activation of input
        :param input_shape: shape of layer input in format (batch size,
                            number of channels, height, width)
        :type input_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width),
                                          not equal (1, 1)
        :param pair of integers stride: stride of max pool
        :param pair of integers padding: padding of max pool
        :returns: Estimated impact of input on output of network
        :rtype: NpInterval
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
                neigh_max_low = np.asarray([-np.inf], dtype=config.floatX)
                neigh_max_upp = np.asarray([-np.inf], dtype=config.floatX)
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

        :param NpInterval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param NpInterval activation: estimated activation of input
        :param input_shape: shape of layer input in format (batch size,
                            number of channels, height, width)
        :type input_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width),
                                          not equal (1, 1)
        :param pair of integers stride: stride of avg pool
        :param pair of integers padding: padding of avg pool
        :returns: Estimated impact of input on output of network
        :rtype: NpInterval
        """
        # n_batches, n_in, h, w - number of batches, number of channels,
        # image height, image width
        n_batches, n_in, h, w = input_shape

        pad_h, pad_w = padding
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
        return self

    def op_d_conv(self, input_shape, filter_shape, weights,
                  stride, padding, n_groups, conv_layer=None):
        """Returns estimated impact of input of convolutional layer on output
        of network.

        :param NpInterval self: estimated impact of output of layer on output
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
        :param conv_layer: convolutional layer in which theano graph might
                           be saved
        :type conv_layer: DerestConvolutionalLayer
        :returns: Estimated impact of input on output of network
        :rtype: NpInterval
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
        if stride == (1, 1):
            # arguments for theano convolution operation
            op_stride = stride
            op_n_batches, op_fh, op_fw = n_batches, fh, fw
            op_n_in = n_out
            op_n_out = n_in
            op_h = h - fh + 1
            op_w = w - fw + 1
            op_pad_h = fh - 1
            op_pad_w = fw - 1
            op_padding = op_pad_h, op_pad_w
            op_image_shape = op_n_batches, op_n_in, op_h, op_w
            op_filter_shape = op_n_out, op_fh, op_fw
            op_n_groups = n_groups
            op_input = output
            op_low = op_input.lower
            op_upp = op_input.upper
            op_weights = np.swapaxes(weights, 0, 1)
            op_g_n_in = op_n_in / op_n_groups
            op_weights = [op_weights[:, i * op_g_n_in:(i + 1) * op_g_n_in,
                                     :, :] for i in xrange(op_n_groups)]
            op_weights = np.concatenate(op_weights, axis=0)
            conv_op_key = (op_image_shape, op_filter_shape, op_padding,
                           op_n_groups)
            if conv_op_key not in conv_layer.theano_ops:
                t_lower, t_upper = T.tensor4(), T.tensor4()
                result_lower, result_upper = self._theano_op_conv(
                    t_lower, t_upper, op_weights, op_image_shape,
                    op_filter_shape, None, op_stride, op_padding, op_n_groups
                )
                op_conv_function = function([t_lower, t_upper],
                                            [result_lower, result_upper])
                conv_layer.theano_ops[conv_op_key] = op_conv_function
            conv_op = conv_layer.theano_ops[conv_op_key]
            lower, upper = conv_op(op_low, op_upp)
            result = NpInterval(lower, upper)
            result = result[:, :, pad_h:(h - pad_h), pad_w:(w - pad_w)]
            return result

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
        np_matrix = np.eye(n_outputs, dtype=config.floatX)
        return NpInterval(np_matrix, np_matrix)
