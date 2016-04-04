"""Intervals implemented in Theano including special functions for
sparsifying.

This module contains Interval class and auxiliary objects.
"""

from numlike import Numlike
from theano import function
from theano import tensor as T
from theano import shared

import numpy

from athenet.utils.misc import convolution, reshape_for_padding as \
    misc_reshape_for_padding

NEUTRAL_INTERVAL_LOWER = 0.0
NEUTRAL_INTERVAL_UPPER = 0.0
NEUTRAL_INTERVAL_VALUES = (NEUTRAL_INTERVAL_LOWER, NEUTRAL_INTERVAL_UPPER)

DEFAULT_INTERVAL_LOWER = 0.0
DEFAULT_INTERVAL_UPPER = 255.0
DEFAULT_INTERVAL_VALUES = (DEFAULT_INTERVAL_LOWER, DEFAULT_INTERVAL_UPPER)


class Interval(Numlike):
    """Theano interval matrix class

    Represents matrix of intervals. Behaves like limited numpy.ndarray of
    intervals.

    .. note:: Should be treated as interval type with bounds as Theano nodes.
              Operations on Interval create nodes in Theano graph. In order to
              read result of given operations, use eval method.
    """

    def __init__(self, lower, upper):
        """Creates interval.

        :param theano tensor lower: lower bound of Interval to be set
        :param theano tensor upper: upper bound of Interval to be set

        .. note:: lower must be lower than upper. It is not being checked.
        """
        super(Interval, self).__init__()
        self.lower = lower
        self.upper = upper

    def __getitem__(self, at):
        """Returns specified slice of interval as a interval.

        :param at: coordinates / slice to be taken
        :rtype: Interval

        .. note:: Does not copy data.
        """
        return Interval(self.lower[at], self.upper[at])

    def __setitem__(self, at, other):
        """Just like Theano set_subtensor function, but as a operator.

        :param at: coordinates / slice to be set
        :param other: data to be put at 'at'
        :type other: Interval
        """
        self.lower = T.set_subtensor(self.lower[at], other.lower)
        self.upper = T.set_subtensor(self.upper[at], other.upper)

    @property
    def shape(self):
        """Returns shape of interval. Checks only 'lower' matrix.

        :rtype: theano tuple of integers

        .. note:: does not require self.upper for computations. Therefore it is
        not safe, but faster.
        """
        return self.lower.shape

    def __add__(self, other):
        """Returns sum of two intervals.

        :param other: matrix to be added
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower + other.lower
            res_upper = self.upper + other.upper
        else:
            res_lower = self.lower + other
            res_upper = self.upper + other
        return Interval(res_lower, res_upper)

    def __sub__(self, other):
        """Returns difference between two intervals.

        :param other: matrix to be subtracted
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            res_lower = self.lower - other.upper
            res_upper = self.upper - other.lower
        else:
            res_lower = self.lower - other
            res_upper = self.upper - other
        return Interval(res_lower, res_upper)

    def __mul__(self, other):
        """Returns product of two intervals.

        :param other: matrix to be multiplied
        :type other: Interval or numpy.ndarray
        :rtype: Interval
        """
        if isinstance(other, Interval):
            ll = self.lower * other.lower
            lu = self.lower * other.upper
            ul = self.upper * other.lower
            uu = self.upper * other.upper
            l = T.minimum(ll, lu)
            l = T.minimum(l, ul)
            l = T.minimum(l, uu)
            u = T.maximum(ll, lu)
            u = T.maximum(u, ul)
            u = T.maximum(u, uu)
            return Interval(l, u)
        else:
            ll = self.lower * other
            uu = self.upper * other
            l = T.minimum(ll, uu)
            u = T.maximum(ll, uu)
            return Interval(l, u)

    def __div__(self, other):
        """Returns quotient of self and other.

        :param other: divisor
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval

        .. warning:: Divisor should not contain zero.
        """
        lower = self.lower
        upper = self.upper
        if isinstance(other, Interval):
            o_lower = other.lower
            o_upper = other.upper
            a = T.switch(T.gt(o_lower, 0.0), 1, 0)  # not(b_la), b_ua
            b = T.switch(T.gt(lower, 0.0), 1, 0)
            c = T.switch(T.gt(upper, 0.0), 1, 0)
            b_lb = T.or_(T.and_(a, b),
                         T.and_(1 - a, c))
            b_ub = T.or_(1 - T.or_(a, b),
                         T.and_(a, 1 - c))
            la = T.switch(a, lower, upper)
            ua = T.switch(a, upper, lower)
            lb = T.switch(b_lb, o_upper, o_lower)
            ub = T.switch(b_ub, o_upper, o_lower)
            return Interval(la / lb, ua / ub)
        else:
            if other > 0:
                return Interval(lower / other, upper / other)
            else:
                return Interval(upper / other, lower / other)

    def __rdiv__(self, other):
        """Returns quotient of other and self.

        :param other: dividend
        :type other: Interval or numpy.ndarray or float
        :rtype: Interval

        .. warning:: Divisor (self) should not contain zero.
        """
        if isinstance(other, Interval):
            # Should never happen. __div__ should be used instead.
            raise NotImplementedError
        else:
            lower = self.lower
            upper = self.upper
            l_o = (other > 0)
            if l_o:
                return Interval(other / upper, other / lower)
            else:
                return Interval(other / lower, other / upper)

    def reciprocal(self):
        """Returns reciprocal of the interval.

        It is a partial reciprocal function. Does not allow 0 to be within
        interval. Should not be treated as general reciprocal function.

        :rtype: Interval
        """
        # Note: Could be considered whether not to use input check.
        # If 0 is within interval, returns 1/0 that, we hope, will throw
        # some exception on the device. Be careful with this.

        # Input check below causes program interrupt if any _has_zero happened.
        # return Interval(switch(self._has_zero(),
        #                       T.constant(1)/T.constant(0),
        #                       T.inv(self.upper)),
        #                T.inv(self.lower))
        return Interval(T.inv(self.upper), T.inv(self.lower))

    def neg(self):
        """For interval [a, b], returns interval [-b, -a]."""
        return Interval(T.neg(self.upper), T.neg(self.lower))

    def exp(self):
        """Returns interval representing the exponential of the interval."""
        return Interval(T.exp(self.lower), T.exp(self.upper))

    def square(self):
        """For interval I, returns I' such that for any x in I, I' contains
        x*x and no other.
        :rtype: Interval

        :Example:

        >>> from athenet.algorithm.numlike import Interval
        >>> import numpy
        >>> a = numpy.array([-1])
        >>> b = numpy.array([1])
        >>> i = Interval(-1, 1)
        >>> s = i.square()
        >>> s.eval()
        >>> (numpy.ndarray([0]), numpy.ndarray([1]))

        """
        lsq = self.lower * self.lower
        usq = self.upper * self.upper
        u = T.maximum(lsq, usq)
        l = T.switch(self._has_zero(), 0, T.minimum(lsq, usq))
        return Interval(l, u)

    def power(self, exponent):
        """For interval i, returns i^exponent.

        :param exponent: Number to be passed as exponent to i^exponent.
        :type exponent: integer or float
        :rtype: Interval

        .. note:: If interval contains some elements lower/equal to 0, exponent
                  should be integer."""
        # If You want to understand what is happening here, make plot of
        # f(x, y) = x^y domain. 'if's divide this domain with respect to
        # monotonicity.
        le = T.pow(self.lower, exponent)
        ue = T.pow(self.upper, exponent)
        if isinstance(exponent, (int, long)):
            if exponent > 0:
                if exponent % 2 == 0:
                    l = T.switch(self._has_zero(), 0, T.minimum(le, ue))
                    u = T.maximum(le, ue)
                else:
                    l = le
                    u = ue
            else:
                if exponent % 2 == 0:
                    l = T.minimum(le, ue)
                    u = T.maximum(le, ue)
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
        return Interval(l, u)

    def dot(self, other):
        """Returns dot product of Interval(self) vector and a number array
        (other).

        :param numpy.ndarray or theano.tensor other: number array to be
                                                     multiplied
        :rtype: Interval
        """
        lower = self.lower
        upper = self.upper
        other_negative = T.minimum(other, 0.0)
        other_positive = T.maximum(other, 0.0)
        lower_pos_dot = T.dot(lower, other_positive)
        lower_neg_dot = T.dot(lower, other_negative)
        upper_pos_dot = T.dot(upper, other_positive)
        upper_neg_dot = T.dot(upper, other_negative)
        res_lower = lower_pos_dot + upper_neg_dot
        res_upper = upper_pos_dot + lower_neg_dot
        return Interval(res_lower, res_upper)

    def max(self, other):
        """Returns interval such that for any numbers (x, y) in a pair of
        corresponding intervals in (self, other) arrays, max(x, y) is in result
        and no other.

        :param other: Interval to be compared
        :type other: Interval or theano.tensor
        :rtype: Interval
        """
        if isinstance(other, Interval):
            return Interval(T.maximum(self.lower, other.lower),
                            T.maximum(self.upper, other.upper))
        else:
            return Interval(T.maximum(self.lower, other),
                            T.maximum(self.upper, other))

    def amax(self, axis=None, keepdims=False):
        """Returns maximum of an Interval along an axis.

        Works like theano.tensor.max.
        :param axis: axis or axes along which to compute the maximum
        :param keepdims: If this is set to True, the axes which are reduced are
                         left in the result as dimensions with size one. With
                         this option, the result will broadcast correctly
                         against the original tensor.
        :type keepdims: boolean
        """
        lower = self.lower.max(axis=axis, keepdims=keepdims)
        upper = self.upper.max(axis=axis, keepdims=keepdims)
        return Interval(lower, upper)

    def reshape(self, shape):
        """Reshapes interval tensor like theano Tensor.

        :param shape: Something that can be converted to a symbolic vector of
                      integers.
        """
        return Interval(self.lower.reshape(shape),
                        self.upper.reshape(shape))

    def flatten(self):
        """Flattens interval tensor like theano Tensor.

        :return: Variable with same dtype as x and outdim dimensions.
        :rtype: Variable with the same shape as x in the leading outdim-1
                dimensions, but with all remaining dimensions of x collapsed
                into the last dimension.
        """
        return Interval(self.lower.flatten(),
                        self.upper.flatten())

    def sum(self, axis=None, dtype=None, keepdims=False):
        """Vector operation like in numpy.ndarray.

        :param integer or None axis: axis along which this function sums
        :param type or None dtype: just like dtype argument in
                                   theano.tensor.sum
        :param Boolean keepdims: Whether to keep squashed dimensions of size 1
        """
        return Interval(self.lower.sum(axis=axis, dtype=dtype,
                                       keepdims=keepdims),
                        self.upper.sum(axis=axis, dtype=dtype,
                                       keepdims=keepdims))

    def abs(self):
        """Returns absolute value of Interval."""
        lower = T.switch(T.gt(self.lower, 0.0), self.lower,
                         T.switch(T.lt(self.upper, 0.0), -self.upper, 0.0))
        upper = T.maximum(-self.lower, self.upper)
        return Interval(lower, upper)

    @property
    def T(self):
        """Vector operation like in numpy.ndarray."""
        return Interval(self.lower.T,
                        self.upper.T)

    @staticmethod
    def from_shape(shp, neutral=True, lower_val=None,
                   upper_val=None):
        """Returns Interval of shape shp with given lower and upper values.

        :param tuple of integers or integer shp : shape of created Interval
        :param Boolean neutral: if True sets (lower_val, upper_val) to
                                NEUTRAL_INTERVAL_VALUES, otherwise to
                                DEFAULT_INTERVAL_VALUES, works only if pair is
                                not set by passing arguments.
        :param float lower_val: value of lower bound
        :param float upper_val: value of upper bound
        """
        if lower_val > upper_val:
            raise ValueError("lower_val > upper_val in newly created Interval")
        if lower_val is None:
            lower_val = NEUTRAL_INTERVAL_LOWER if neutral else \
                        DEFAULT_INTERVAL_LOWER
        if upper_val is None:
            upper_val = NEUTRAL_INTERVAL_UPPER if neutral else \
                        DEFAULT_INTERVAL_UPPER
        lower_array = numpy.ndarray(shp)
        upper_array = numpy.ndarray(shp)
        lower_array.fill(lower_val)
        upper_array.fill(upper_val)
        lower = shared(lower_array)
        upper = shared(upper_array)
        return Interval(lower, upper)

    def reshape_for_padding(self, shape, padding, lower_val=None,
                            upper_val=None):
        """Returns padded Interval.

        :param tuple of 4 integers shape: shape of input in format
                                          (batch size, number of channels,
                                           height, width)
        :param pair of integers padding: padding to be applied
        :param float lower_val: value of lower bound in new fields
        :param float upper_val: value of upper bound in new fields
        :returns: padded layer_input
        :rtype: Interval
        """
        if lower_val is None:
            lower_val = NEUTRAL_INTERVAL_UPPER
        if upper_val is None:
            upper_val = NEUTRAL_INTERVAL_UPPER
        n_batches, n_in, h, w = shape
        padded_low = misc_reshape_for_padding(self.lower, (h, w, n_in),
                                              n_batches, padding, lower_val)
        padded_upp = misc_reshape_for_padding(self.upper, (h, w, n_in),
                                              n_batches, padding, upper_val)
        return Interval(padded_low, padded_upp)

    def eval(self, eval_map=None):
        """Evaluates interval in terms of theano TensorType eval method.

        :param eval_map: map of Theano variables to be set, just like in
                          theano.tensor.dtensorX.eval method
        :type eval_map: {theano.tensor: numpy.ndarray} dict

        :rtype: (lower, upper) pair of numpy.ndarrays
        """
        has_args = eval_map is not None
        if has_args:
            has_args = len(eval_map) != 0
        if not has_args:
            try:
                f = function([], [self.lower, self.upper])
                rlower, rupper = f()
                return rlower, rupper
            except:
                return self.lower, self.upper
        keys = eval_map.keys()
        values = eval_map.values()
        f = function(keys, [self.lower, self.upper])
        rlower, rupper = f(*values)
        return rlower, rupper

    def op_relu(self):
        """Returns result of relu operation on given Interval.

        :rtype: Interval
        """
        lower = T.maximum(self.lower, 0.0)
        upper = T.maximum(self.upper, 0.0)
        return Interval(lower, upper)

    def op_softmax(self, input_shp):
        """Returns result of softmax operation on given Interval.

        :param integer input_shp: shape of 1D input
        :rtype: Interval

        .. note:: Implementation note. Tricks for encountering representation
                  problems:
                  Theoretically, softmax(input) == softmax(input.map(x->x+c))
                  for Real x, y. For floating point arithmetic it is not true.
                  e.g. in expression:

                  e^x / (e^x + e^y) = 0.0f / (0.0f + 0.0f) = NaN for too little
                  values of x, y
                  or
                  e^x / (e^x + e^y) = +Inf / +Inf = NaN for too hight values of
                  x, y.
                  There is used a workaround:
                      * _low endings are for softmax with variables shifted so
                        that input[i].upper() == 0
                      * _upp endings are for softmax with variables shifted so
                        that input[i].lower() == 0
        """
        result = Interval.from_shape(input_shp, neutral=True)
        for i in xrange(input_shp):
            input_low = (self - self.upper[i]).exp()
            input_upp = (self - self.lower[i]).exp()
            sum_low = Interval.from_shape(1, neutral=True)
            sum_upp = Interval.from_shape(1, neutral=True)
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
            result[i] = Interval(lower_upp, upper_low)
        return result

    def op_norm(self, input_shape, local_range, k, alpha, beta):
        """Returns estimated activation of LRN layer.

        :param input_shape: shape of Interval in format
                            (n_channels, height, width)
        :param integer local_range: size of local range in local range
                                    normalization
        :param integer k: local range normalization k argument
        :param integer alpha: local range normalization alpha argument
        :param integer beta: local range normalization beta argument
        :type input_shape: tuple of 3 integers
        :rtype: Interval
        """
        k_array = numpy.array([k])
        alpha_array = numpy.array([alpha])
        lower = self.lower
        upper = self.upper
        half = local_range / 2
        sq = self.square()
        n_channels, h, w = input_shape
        extra_channels = self.from_shape((n_channels + 2 * half, h, w),
                                         neutral=True)
        extra_channels[half:half + n_channels, :, :] = sq
        neigh_sums = self.from_shape(input_shape, neutral=True)

        for i in xrange(local_range):
            if i != half:
                neigh_sums += extra_channels[i:i + n_channels, :, :]
        c1 = neigh_sums * alpha_array + k_array
        c2 = alpha_array
        extreme = c1 * numpy.array([2.0]) - sq * c2
        upper_v = T.sqrt(c1.lower * 2.0 / alpha)
        lower_alpha = alpha * sq.lower
        upper_alpha = alpha * sq.upper
        lower1 = lower / T.pow(c1.upper + lower_alpha, beta)
        lower2 = upper / T.pow(c1.upper + upper_alpha, beta)
        upper1 = upper / T.pow(c1.lower + upper_alpha, beta)
        upper2 = lower / T.pow(c1.lower + lower_alpha, beta)
        res_lower = T.minimum(lower1, lower2)
        res_upper = T.switch(extreme._has_zero(), upper_v,
                             T.minimum(upper1, upper2))
        return Interval(res_lower, res_upper)

    def op_conv(self, weights, image_shape, filter_shape, biases, stride,
                padding, n_groups):
        """Returns estimated activation of convolution applied to Interval.

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
        :type weights: theano.tensor3
        :type filter_shape: tuple of 3 integers
        :type biases: theano.vector
        :type stride: pair of integers
        :type padding: pair of integers
        :type n_groups: integer
        :rtype: Interval
        """
        image_shape = (image_shape[1], image_shape[2], image_shape[0])
        filter_shape = (filter_shape[1], filter_shape[2], filter_shape[0])
        args = (stride, n_groups, image_shape, padding, 1, filter_shape)
        input_lower = self.lower.dimshuffle('x', 0, 1, 2)
        input_upper = self.upper.dimshuffle('x', 0, 1, 2)
        input_lower_padded = misc_reshape_for_padding(input_lower, image_shape,
                                                      1, padding)
        input_upper_padded = misc_reshape_for_padding(input_upper, image_shape,
                                                      1, padding)
        weights_positive = T.maximum(weights, 0.)
        weights_negative = T.minimum(weights, 0.)
        conv_lower_positive = convolution(input_lower_padded, weights_positive,
                                          *args)
        conv_lower_negative = convolution(input_lower_padded, weights_negative,
                                          *args)
        conv_upper_positive = convolution(input_upper_padded, weights_positive,
                                          *args)
        conv_upper_negative = convolution(input_upper_padded, weights_negative,
                                          *args)
        conv_result_lower = conv_lower_positive + conv_upper_negative
        conv_result_upper = conv_lower_negative + conv_upper_positive
        _, n_in, h, w = conv_result_lower.shape
        conv_result_lower_3d = conv_result_lower.reshape((n_in, h, w))
        conv_result_upper_3d = conv_result_upper.reshape((n_in, h, w))
        result_interval = Interval(conv_result_lower_3d, conv_result_upper_3d)
        return result_interval + biases.dimshuffle(0, 'x', 'x')

    def op_d_relu(self, activation):
        """Returns estimated impact of relu layer on output of network.

        :param Interval activation: activation of relu layer
        :returns: Impact of input of relu on output of network
        :rtype: Interval
        """
        out_lower = self.lower
        out_upper = self.upper
        act_low = activation.lower
        act_upp = activation.upper
        low_gt_zero = T.gt(act_low, 0.0)
        upp_lt_zero = T.lt(act_upp, 0.0)
        lower = T.switch(low_gt_zero, out_lower,
                         T.switch(upp_lt_zero, 0.0, T.minimum(out_lower, 0.0)))
        upper = T.switch(low_gt_zero, out_upper,
                         T.switch(upp_lt_zero, 0.0, T.maximum(out_upper, 0.0)))
        return Interval(lower, upper)

    def op_d_max_pool(self, activation, activation_shape, poolsize, stride,
                      padding):
        """Returns estimated impact of max pool layer on output of network.

        :param Interval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param Interval activation: estimated activation of input
        :param activation_shape: shape of activation in format (batch size,
                                 number of channels, height, width)
        :type activation_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width),
                                          not equal (1, 1)
        :param pair of integers stride: stride of max pool
        :param pair of integers padding: padding of max pool
        :returns: Estimated impact of input on output of network
        :rtype: Interval
        """
        n_batches, n_in, h, w = activation_shape
        pad_h, pad_w = padding
        activation = activation.reshape_for_padding(activation_shape, padding,
                                                    lower_val=-numpy.inf,
                                                    upper_val=-numpy.inf)
        activation_shape = (n_batches, n_in, h + 2 * pad_h, w + 2 * pad_w)
        h += 2 * pad_h
        w += 2 * pad_w
        # n_batches, n_in, h, w - number of batches, number of channels,
        #                         image height, image width
        # fh, fw - pool height, pool width
        fh, fw = poolsize
        stride_h, stride_w = stride
        output = self
        result = activation.from_shape(activation_shape, neutral=True)
        for at_h in xrange(0, h - fh + 1, stride_h):
            # at_out_h - height of output corresponding to pool at position at
            # h
            at_out_h = at_h / stride_h
            for at_w in xrange(0, w - fw + 1, stride_w):
                # at_out_w - height of output corresponding to pool at
                # position at_w
                at_out_w = at_w / stride_w
                # any input on any filter frame
                for at_f_h in xrange(at_h, at_h + fh):
                    for at_f_w in xrange(at_w, at_w + fw):
                        # maximum lower and upper of neighbours
                        neigh_max_low = shared(-numpy.inf)
                        neigh_max_upp = shared(-numpy.inf)
                        neigh_max_itv = Interval(neigh_max_low, neigh_max_upp)
                        act_slice = activation[:, :, at_f_h, at_f_w]
                        # setting maximum lower and upper of neighbours
                        for at_f_h_neigh in xrange(at_h, at_h + fh):
                            for at_f_w_neigh in xrange(at_w, at_w + fw):
                                if (at_f_h_neigh, at_f_w_neigh) != (
                                        at_f_h, at_f_w):
                                    neigh_slice = activation[:, :,
                                                             at_f_h_neigh,
                                                             at_f_w_neigh]
                                    neigh_max_itv = \
                                        neigh_max_itv.max(neigh_slice)
                        # must have impact on output
                        low_gt_neigh_max_upp = \
                            T.gt(act_slice.lower, neigh_max_itv.upper)
                        # cannot have impact on output
                        upp_gt_neigh_max_low = \
                            T.gt(act_slice.upper, neigh_max_itv.lower)
                        # might have impact on output
                        output_slice = output[:, :, at_out_h, at_out_w]
                        mixed_low = T.minimum(output_slice.lower, 0.0)
                        mixed_upp = T.maximum(output_slice.upper, 0.0)
                        to_add_low = T.switch(
                            low_gt_neigh_max_upp,
                            output_slice.lower,
                            T.switch(
                                upp_gt_neigh_max_low,
                                mixed_low,
                                T.zeros_like(mixed_low)
                            )
                        )
                        to_add_upp = T.switch(
                            low_gt_neigh_max_upp,
                            output_slice.upper,
                            T.switch(
                                upp_gt_neigh_max_low,
                                mixed_upp,
                                T.zeros_like(mixed_upp)
                            )
                        )
                        itv_to_add = Interval(to_add_low, to_add_upp)
                        result[:, :, at_f_h, at_f_w] = \
                            result[:, :, at_f_h, at_f_w] + itv_to_add

        return result[:, :, pad_h:h - pad_h, pad_w:w - pad_w]

    def op_d_avg_pool(self, activation, activation_shape, poolsize, stride,
                      padding):
        """Returns estimated impact of avg pool layer on output of network.

        :param Interval self: estimated impact of output of layer on output
                               of network in shape (batch_size, number of
                               channels, height, width)
        :param Interval activation: estimated activation of input
        :param activation_shape: shape of activation in format (batch size,
                                 number of channels, height, width)
        :type activation_shape: tuple of 4 integers
        :param pair of integers poolsize: pool size in format (height, width)
        :param pair of integers stride: stride of max pool
        :param pair of integers padding: padding of avg pool
        :returns: Estimated impact of input on output of network
        :rtype: Interval
        """
        n_batches, n_in, h, w = activation_shape
        pad_h, pad_w = padding
        activation = activation.reshape_for_padding(activation_shape, padding,
                                                    lower_val=-numpy.inf,
                                                    upper_val=-numpy.inf)
        activation_shape = (n_batches, n_in, h + 2 * pad_h, w + 2 * pad_w)
        h += 2 * pad_h
        w += 2 * pad_w
        # n_batches, n_in, h, w - number of batches, number of channels,
        #                         image height, image width
        # fh, fw - pool height, pool width
        fh, fw = poolsize
        stride_h, stride_w = stride
        output = self
        result = activation.from_shape(activation_shape, neutral=True)
        return result[:, :, pad_h:h - pad_h, pad_w:w - pad_w]

    @staticmethod
    def derest_output(n_outputs):
        """Generates Interval of impact of output on output.

        :param int n_outputs: Number of outputs of network.
        :returns: 2D square Interval in shape (n_batches, n_outputs) with one
                  different "1" in every batch, like numpy.eye(n_outputs)
        :rtype: Interval
        """
        np_matrix = numpy.eye(n_outputs)
        th_matrix = shared(np_matrix)
        return Interval(th_matrix, th_matrix)

    def __repr__(self):
        """Standard repr method."""
        return '[' + repr(self.lower) + ', ' + repr(self.upper) + ']'

    def __str__(self):
        """"Standard str method."""
        return '[' + str(self.lower) + ', ' + str(self.upper) + ']'

    def _has_zero(self):
        """For any interval in Interval, returns whether is contains zero.

        :rtype: Boolean
        """
        return T.and_(T.lt(self.lower, 0.0), T.gt(self.upper, 0.0))
