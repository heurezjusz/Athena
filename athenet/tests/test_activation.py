"""Testing athenet.algorithm.derest.activation functions.
"""

import numpy as np
import theano
import theano.tensor as T
import unittest
from math import e
from nose.tools import assert_almost_equal, assert_greater
from numpy.testing import assert_array_almost_equal

from athenet.algorithm.derest.layers.fully_connected import a_fully_connected
from athenet.algorithm.derest.layers.convolutional import a_conv
from athenet.algorithm.derest.layers.pool import a_pool
from athenet.algorithm.derest.layers.dropout import a_dropout
from athenet.algorithm.derest.layers.norm import a_norm
from athenet.algorithm.derest.layers.relu import a_relu
from athenet.algorithm.derest.layers.softmax import a_softmax

from athenet.algorithm.numlike import TheanoInterval, Nplike

theano.config.exception_verbosity = 'high'


def array_almost_equal(x, y):
    if theano.config.floatX == 'float32':
        return assert_array_almost_equal(x, y, decimal=3)
    else:
        return assert_array_almost_equal(x, y)


def A(x):
    return np.array(x, dtype=theano.config.floatX)


def nplike(x):
    return Nplike(A(x))


class ActivationTest(unittest.TestCase):

    def prepare(self):
        self.v = np.arange(24, dtype=theano.config.floatX) + 3.0
        self.at_v = 0
        return self.s, self.v, self.make_arr

    def s(self):
        if self.at_v >= len(self.v):
            raise ValueError
        ret = self.v[self.at_v]
        self.at_v += 1
        return ret

    def make_arr(self, shp):
        sz = np.prod(shp)
        a = np.ndarray(sz)
        for i in range(sz):
            a[i] = self.s()
        return a.reshape(shp)


class FullyConnectedActivationTest(ActivationTest):

    def test_1D_simple(self):
        res = a_fully_connected(nplike([1]), A([2]), A([0]))
        array_almost_equal(res.eval(), A([2]))

    def test_2D_simple_used_1D_of_weights(self):
        s, v, m = self.prepare()
        inp, w, b = m(2), m(2), A([1.0])
        res = a_fully_connected(inp, w, b)
        array_almost_equal(res, A([v[0] * v[2] + v[1] * v[3] + 1.0]))

    def test_2D_simple_used_2D_of_weights(self):
        s, v, m = self.prepare()
        inp = m(1)
        w = m((1, 2))
        b = m(2)
        array_almost_equal(a_fully_connected(inp, w, b), A([v[0] * v[1] + v[3],
                           v[0] * v[2] + v[4]]))

    def test_2D_simple(self):
        s, v, m = self.prepare()
        inp = m(2)
        w = m((2, 2))
        b = m(2)
        array_almost_equal(a_fully_connected(inp, w, b),
                           A([v[0] * v[2] + v[1] * v[4] + v[6],
                              v[0] * v[3] + v[1] * v[5] + v[7]]))

    def test_2D_2(self):
        s, v, m = self.prepare()
        inp = m(4)
        w = m((4, 2))
        b = m(2)
        rl = v[0] * v[4] + v[1] * v[6] + v[2] * v[8] + v[3] * v[10] + v[12]
        ru = v[0] * v[5] + v[1] * v[7] + v[2] * v[9] + v[3] * v[11] + v[13]
        array_almost_equal(a_fully_connected(inp, w, b), A([rl, ru]))

    def test_3D_using_intervals(self):
        s, v, m = self.prepare()
        inpl = m(2)
        inpu = m(2)
        w = m((2, 2))
        b = A([1, 3])
        crl = A([v[0] * v[4] + v[1] * v[6] + 1,
                 v[0] * v[5] + v[1] * v[7] + 3])
        cru = A([v[2] * v[4] + v[3] * v[6] + 1,
                 v[2] * v[5] + v[3] * v[7] + 3])
        tinpl, tinpu = T.dvectors('inpl', 'inpu')

        iinp = TheanoInterval(tinpl, tinpu)
        res = a_fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        array_almost_equal(rl, crl)
        array_almost_equal(ru, cru)

    def test_3D_negative_weights_using_intervals(self):
        s, v, m = self.prepare()
        inpl = A([[[v[0], v[1]]]])
        inpu = A([[[v[2], v[3]]]])
        w = A([[v[4], -v[5], v[6]], [v[7], v[8], v[9]]])
        b = A([1, 3, 5])
        crl = A([v[0] * v[4] + v[1] * v[7] + 1,
                 v[2] * -v[5] + v[1] * v[8] + 3,
                 v[0] * v[6] + v[1] * v[9] + 5])
        cru = A([v[2] * v[4] + v[3] * v[7] + 1,
                 v[0] * -v[5] + v[3] * v[8] + 3,
                 v[2] * v[6] + v[3] * v[9] + 5])
        tinpl, tinpu = T.tensor3s('inpl', 'inpu')
        iinp = TheanoInterval(tinpl, tinpu)
        res = a_fully_connected(iinp, w, b)
        d = {tinpl: inpl, tinpu: inpu}
        (rl, ru) = res.eval(d)
        array_almost_equal(rl, crl)
        array_almost_equal(ru, cru)

    def test_negative(self):
        inp = nplike([1, -1])
        w = A([[1, 1], [1, -1]])
        b = A([0, 0])
        res = a_fully_connected(inp, w, b)
        c = A([0, 2])
        array_almost_equal(res.eval(), c)


class ConvolutionalActivationTest(ActivationTest):

    def test_trivial(self):
        inp = nplike(A([[[1]]]))
        w = nplike(A([[[[2]]]]))
        b = nplike(A([3]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(inp, inp.shape, w, f_shp, b)
        array_almost_equal(res.eval(), A([[[5]]]))

    def test_1_channel_input_1_conv_feature(self):
        inp = nplike(A([[[0, 0], [2, 3]]]))
        w = nplike(A([[[[7, 5], [3, 2]]]]))
        b = nplike(A([4]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(inp, inp.shape, w, f_shp, b)
        array_almost_equal(res.eval(), A([[[35]]]))

    def test_1_channel_input_1_conv_feature2(self):
        inp = nplike(np.zeros((1, 3, 3), dtype=theano.config.floatX))
        inp[0, 2, 1] = nplike(2)
        inp[0, 2, 2] = nplike(3)
        w = nplike([[[[7, 5], [3, 2]]]])
        b = nplike(A([4]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(inp, inp.shape, w, f_shp, b)
        array_almost_equal(res.eval(), A([[[4, 4], [18, 35]]]))

    def test_use_all_dims(self):
        inp = nplike(np.zeros((2, 4, 4), dtype=theano.config.floatX))
        inp[0:2, 1:3, 1:3] = nplike(A([[[2, 3], [5, 7]],
                                      [[0.2, 0.3], [0.5, 0.7]]]))
        w = nplike(np.zeros((2, 2, 3, 3), dtype=theano.config.floatX))
        w[0, 0, 0, 0] = nplike(1.0)
        w[0, 0, 0, 2] = nplike(2.0)
        w[0, 0, 2, 0] = nplike(3.0)
        w[0, 0, 1, 1] = nplike(4.0)
        w[1, 0, 0, 0] = nplike(5.0)
        w[1, 0, 0, 2] = nplike(6.0)
        w[1, 0, 2, 0] = nplike(7.0)
        w[1, 0, 2, 2] = nplike(8.0)
        w[0, 1, 1, 1] = nplike(9.0)
        w[0, 1, 1, 2] = nplike(10.0)
        w[0, 1, 2, 1] = nplike(11.0)
        w[0, 1, 2, 2] = nplike(12.0)
        w[1, 1, 0, 0] = nplike(13.0)
        w[1, 1, 2, 0] = nplike(14.0)
        w[1, 1, 0, 1] = nplike(15.0)
        w[1, 1, 2, 1] = nplike(16.0)
        w_flipped = w[:, :, ::-1, ::-1]
        b = nplike(A([[[13]], [[23]]]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(inp, inp.shape, w_flipped, f_shp, b)
        array_almost_equal(res.eval(), A([[[39.7, 50.4], [50.5, 49.3]],
                                          [[87.0, 76.2], [44.0, 40.1]]]))

    def test_padding(self):
        inp = nplike(A([[[2, 3], [5, 7]],
                       [[0.2, 0.3], [0.5, 0.7]]]))
        w = nplike(np.zeros((2, 2, 3, 3), dtype=theano.config.floatX))
        w[0, 0, 0, 0] = nplike(1.0)
        w[0, 0, 0, 2] = nplike(2.0)
        w[0, 0, 2, 0] = nplike(3.0)
        w[0, 0, 1, 1] = nplike(4.0)
        w[1, 0, 0, 0] = nplike(5.0)
        w[1, 0, 0, 2] = nplike(6.0)
        w[1, 0, 2, 0] = nplike(7.0)
        w[1, 0, 2, 2] = nplike(8.0)
        w[0, 1, 1, 1] = nplike(9.0)
        w[0, 1, 1, 2] = nplike(10.0)
        w[0, 1, 2, 1] = nplike(11.0)
        w[0, 1, 2, 2] = nplike(12.0)
        w[1, 1, 0, 0] = nplike(13.0)
        w[1, 1, 2, 0] = nplike(14.0)
        w[1, 1, 0, 1] = nplike(15.0)
        w[1, 1, 2, 1] = nplike(16.0)
        w_flipped = w[:, :, ::-1, ::-1]
        b = nplike(A([[[13]], [[23]]]))
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(inp, inp.shape, w_flipped, f_shp, b, padding=(1, 1))
        array_almost_equal(res.eval(), A([[[39.7, 50.4], [50.5, 49.3]],
                                          [[87.0, 76.2], [44.0, 40.1]]]))

    def test_stride(self):
        inp = nplike([[[2, 3], [5, 7]]])
        w = nplike([[[[1, 2], [3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = nplike([[[0]]])
        res = a_conv(inp, inp.shape, w_flipped, f_shp, b, padding=(1, 1),
                   stride=(2, 2))
        array_almost_equal(res.eval(), A([[[8, 9], [10, 7]]]))

    def test_interval_simple(self):
        inpl = A([[[-1, 3], [4, 7]]])
        inpu = A([[[2, 3], [5, 9]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        w = A([[[[1, 2], [-3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        tw = theano.shared(w_flipped, borrow=True)
        b = A([0])
        tb = theano.shared(b, borrow=True)
        inp_shape = (1, 2, 2)
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(iinp, inp_shape, tw, f_shp, tb)
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        array_almost_equal(rl, A([[[18]]]))
        array_almost_equal(ru, A([[[32]]]))

    def test_interval_3x3(self):
        inpl = A([[[-1, 3], [4, 7]]])
        inpu = A([[[2, 3], [5, 9]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        w = A([[[[1, 2], [-3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        tw = theano.shared(w_flipped, borrow=True)
        b = A([0])
        tb = theano.shared(b, borrow=True).dimshuffle(0, 'x', 'x')
        inp_shape = (1, 2, 2)
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        res = a_conv(iinp, inp_shape, tw, f_shp, tb, padding=(1, 1))
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        array_almost_equal(rl, A([[[-4, 6, -9], [14, 18, -24], [8, 18, 7]]]))
        array_almost_equal(ru, A([[[8, 15, -9], [24, 32, -18], [10, 23, 9]]]))

    def test_group_1_in_2_out(self):
        inp = nplike([[[2, 3], [5, 7]], [[2, 3], [5, 7]]])
        w = nplike([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]],
                    [[[5, 6], [7, 8]]], [[[1, 2], [3, 4]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = nplike([[[0]]])
        res = a_conv(inp, inp.shape, w_flipped, f_shp, b, n_groups=2)
        array_almost_equal(res.eval(), A([[[51.0]], [[119.0]],
                                          [[119.0]], [[51.0]]]))

    def test_group_2_in_1_out(self):
        inp = nplike([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                      [[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        w = nplike([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[3, 4], [5, 6]], [[7, 8], [9, 1]]]])
        w_flipped = w[:, :, ::-1, ::-1]
        f_shp = (w.shape[0], w.shape[2], w.shape[3])
        b = nplike([[[0]]])
        res = a_conv(inp, inp.shape, w_flipped, f_shp, b, n_groups=2)
        array_almost_equal(res.eval(), A([[[204.0]], [[247.0]]]))


class PoolActivationTest(ActivationTest):

    def test_simple(self):
        inp = nplike([[[1, 2], [3, 4]]])
        resmax = a_pool(inp, inp.shape, (1, 1), mode="max")
        resavg = a_pool(inp, inp.shape, (1, 1), mode="avg")
        array_almost_equal(resmax.eval(), inp.eval())
        array_almost_equal(resavg.eval(), inp.eval())

    def test_simple2(self):
        inp = Nplike(A([[[1, 2], [3, 4]]]))
        resmax = a_pool(inp, inp.shape, (2, 2), mode="max")
        resavg = a_pool(inp, inp.shape, (2, 2), mode="avg")
        array_almost_equal(resmax.eval(), A([[[4.0]]]))
        array_almost_equal(resavg.eval(), A([[[2.5]]]))

    def test_2D1(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))
        resmax = a_pool(inp, inp.shape, (2, 2), mode="max")
        resavg = a_pool(inp, inp.shape, (2, 2), mode="avg")
        array_almost_equal(resmax.eval(), A([[[5.0, 6.0], [8.0, 9.0]]]))
        array_almost_equal(resavg.eval(), A([[[3.0, 4.0], [6.0, 7.0]]]))

    def test_2D2(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))
        resmax = a_pool(inp, inp.shape, (3, 3), mode="max")
        resavg = a_pool(inp, inp.shape, (3, 3), mode="avg")
        array_almost_equal(resmax.eval(), A([[[9.0]]]))
        array_almost_equal(resavg.eval(), A([[[5.0]]]))

    def test_3D(self):
        inp = Nplike(A([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[2, 3, 4], [5, 6, 7], [8, 9, 1]]]))
        resmax = a_pool(inp, inp.shape, (2, 2), mode="max")
        resavg = a_pool(inp, inp.shape, (2, 2), mode="avg")
        array_almost_equal(resmax.eval(), A([[[5.0, 6.0], [8.0, 9.0]],
                                             [[6.0, 7.0], [9.0, 9.0]]]))
        array_almost_equal(resavg.eval(), A([[[3.0, 4.0], [6.0, 7.0]],
                                             [[4.0, 5.0], [7.0, 5.75]]]))

    def test_3D_interval(self):
        inpl = A([[[-1, 2, 3], [4, 5, 6], [7, -3, 0]],
                  [[2, 3, 4], [5, 6, 7], [8, 9, 1]]])
        inpu = A([[[1, 3, 4], [7, 5, 6], [7, 9, 9]],
                  [[2, 3, 4], [5, 6, 7], [8, 9, 1]]])
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        resmax = a_pool(iinp, (2, 3, 3), (2, 2), mode="max")
        resavg = a_pool(iinp, (2, 3, 3), (2, 2), mode="avg")
        d = {tinpl: inpl, tinpu: inpu}
        rlmax, rumax = resmax.eval(d)
        rlavg, ruavg = resavg.eval(d)
        array_almost_equal(rlmax, A([[[5, 6], [7, 6]], [[6, 7], [9, 9]]]))
        array_almost_equal(rumax, A([[[7, 6], [9, 9]], [[6, 7], [9, 9]]]))
        array_almost_equal(rlavg, A([[[10, 16], [13, 8]],
                                     [[16, 20], [28, 23]]]) / 4.0)
        array_almost_equal(ruavg, A([[[16, 18], [28, 29]],
                                     [[16, 20], [28, 23]]]) / 4.0)


class SoftmaxActivationTest(ActivationTest):

    def test_simple(self):
        inp = nplike([1, 2, 3])
        res = softmax(inp, 3)
        s = e * (1 + e * (1 + e))
        array_almost_equal(res.eval(), A([e / s, e ** 2 / s, e ** 3 / s]))

    def test_corner_cases(self):
        inps = [nplike([1]), nplike([2]), nplike([1, 1]), nplike([0]),
                nplike([0, 0])]
        ress = [a_softmax(inp, 5) for inp in inps]
        cress = [1, 1, 0.5, 1, 0.5]
        for (cres, res) in zip(cress, ress):
            array_almost_equal(res.eval(), cres)

    def test_interval_flat(self):
        inp = A([1, 2, 3, 4, 5])
        tinp = T.dvector('tinp')
        itv = TheanoInterval(tinp, tinp)
        res = a_softmax(itv, 5)
        d = {tinp: inp}
        l, u = res.eval(d)
        array_almost_equal(l, u)
        for i in xrange(5):
            assert_almost_equal(l[i], u[i])
        for i in xrange(4):
            assert_greater(l[i + 1], l[i])

    def test_uniform_input(self):
        inp = np.ones(4, dtype=theano.config.floatX) * 4
        tinp = T.dvector('tinp')
        itv = TheanoInterval(tinp, tinp)
        res = a_softmax(itv, 4)
        d = {tinp: inp}
        l, u = res.eval(d)
        array_almost_equal(l, u)
        for i in xrange(4):
            assert_almost_equal(l[i], u[i])
        for i in xrange(3):
            assert_almost_equal(l[i], l[i + 1])

    def test_one_big_elt(self):
        inp = -20 * np.ones(4, dtype=theano.config.floatX)
        inp[0] = 1
        tinp = T.dvector('tinp')
        itv = TheanoInterval(tinp, tinp)
        res = a_softmax(itv, 4)
        d = {tinp: inp}
        l, u = res.eval(d)
        array_almost_equal(l, u)
        assert_almost_equal(l[0], 1)
        assert_almost_equal(l[1], 0)
        assert_almost_equal(l[2], 0)
        assert_almost_equal(l[3], 0)

    def test_case1(self):
        tinpl, tinpu = T.dvectors('tinpl', 'tinpu')
        itv = TheanoInterval(tinpl, tinpu)
        res = a_softmax(itv, 3)
        inp = A([1, 2, 3])
        d = {tinpl: inp, tinpu: inp}
        l, u = res.eval(d)
        cres = (e - 1) / (e ** 3 - 1)
        array_almost_equal(l, u)
        array_almost_equal(l, A([cres, cres * e, cres * e ** 2]))

    def test_case2(self):
        tinpl, tinpu = T.dvectors('tinpl', 'tinpu')
        itv = TheanoInterval(tinpl, tinpu)
        res = a_softmax(itv, 2)
        inpl = A([1, -1])
        inpu = A([2, 3])
        d = {tinpl: inpl, tinpu: inpu}
        l, u = res.eval(d)

        def calc_cres(e1, e2):
            return (e ** e1) / (e ** e1 + e ** e2)

        cresl = A([calc_cres(a, b) for (a, b) in
                   [(1, 3), (-1, 2)]])
        cresu = A([calc_cres(a, b) for (a, b) in
                   [(2, -1), (3, 1)]])
        array_almost_equal(l, cresl)
        array_almost_equal(u, cresu)

    def test_case3(self):
        tinpl, tinpu = T.dvectors('tinpl', 'tinpu')
        itv = TheanoInterval(tinpl, tinpu)
        res = a_softmax(itv, 3)

        inpl = A([1, -1, 2])
        inpu = A([2, 3, 4])
        d = {tinpl: inpl, tinpu: inpu}
        l, u = res.eval(d)

        def calc_cres(e1, e2, e3):
            return (e ** e1) / (e ** e1 + e ** e2 + e ** e3)

        cresl = A([calc_cres(a, b, c) for (a, b, c) in
                   [(1, 3, 4), (-1, 2, 4), (2, 2, 3)]])
        cresu = A([calc_cres(a, b, c) for (a, b, c) in
                   [(2, -1, 2), (3, 1, 2), (4, 1, -1)]])
        array_almost_equal(l, cresl)
        array_almost_equal(u, cresu)

    def test_best_worst_case_for_specific_interval(self):
        tinpl, tinpu = T.dvectors('tinpl', 'tinpu')
        itv = TheanoInterval(tinpl, tinpu)
        res = a_softmax(itv, 4)

        inpl1 = np.ones(4, dtype=theano.config.floatX) + 1
        inpu1 = inpl1 + 1
        d1 = {tinpl: inpl1, tinpu: inpu1}
        l1, u1 = res.eval(d1)
        inp2 = np.ones(4, dtype=theano.config.floatX) * 2.0
        inp2[0] = 1.0
        d2 = {tinpl: inp2, tinpu: inp2}
        l2, u2 = res.eval(d2)
        inp3 = 2.0 - inp2
        d3 = {tinpl: inp3, tinpu: inp3}
        l3, u3 = res.eval(d3)
        d1 = l1[0] - l2[0]
        d2 = u1[0] - u3[0]
        assert_greater(0.01, abs(d1))
        assert_greater(0.01, abs(d2))


class NormActivationTest(ActivationTest):

    def test_case1(self):
        inp = nplike([[[1, 10], [100, 1000]]])
        out = a_norm(inp, (1, 2, 2))
        array_almost_equal(out.eval(), A([[[0.9999850, 9.9850262],
                                           [87.2195949, 101.9378639]]]))


    def test_case1_interval(self):
        inp = A([[[1, 10], [100, 1000]]])
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        out = a_norm(iinp, (1, 2, 2))

        d = {tinpl: inp, tinpu: inp}
        l, u = out.eval(d)
        array_almost_equal(l, u)
        array_almost_equal(l, A([[[0.9999850, 9.9850262],
                                  [87.2195949, 101.9378639]]]))

    def test_case2(self):
        inp1 = nplike([[[1]]])
        inp2 = nplike([[[10]]])
        inp3 = nplike([[[100]]])
        inp4 = nplike([[[1000]]])
        out1 = a_norm(inp1, (1, 1, 1))
        out2 = a_norm(inp2, (1, 1, 1))
        out3 = a_norm(inp3, (1, 1, 1))
        out4 = a_norm(inp4, (1, 1, 1))

        res1 = out1.eval()
        res2 = out2.eval()
        res3 = out3.eval()
        res4 = out4.eval()
        array_almost_equal(res1, A([[[0.9999850]]]))
        array_almost_equal(res2, A([[[9.9850262]]]))
        array_almost_equal(res3, A([[[87.2195949]]]))
        array_almost_equal(res4, A([[[101.9378639]]]))

    def test_case2_interval(self):
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        out = a_norm(iinp, (1, 1, 1))

        inp1 = A([[[1]]])
        inp2 = A([[[10]]])
        inp3 = A([[[100]]])
        inp4 = A([[[1000]]])
        d1 = {tinpl: inp1, tinpu: inp1}
        d2 = {tinpl: inp2, tinpu: inp2}
        d3 = {tinpl: inp3, tinpu: inp3}
        d4 = {tinpl: inp4, tinpu: inp4}
        l1, u1 = out.eval(d1)
        l2, u2 = out.eval(d2)
        l3, u3 = out.eval(d3)
        l4, u4 = out.eval(d4)
        array_almost_equal(l1, u1)
        array_almost_equal(l2, u2)
        array_almost_equal(l3, u3)
        array_almost_equal(l4, u4)
        array_almost_equal(l1, A([[[0.9999850]]]))
        array_almost_equal(l2, A([[[9.9850262]]]))
        array_almost_equal(l3, A([[[87.2195949]]]))
        array_almost_equal(l4, A([[[101.9378639]]]))

    def test_case3(self):
        inp1 = nplike([[[1]], [[1]]])
        inp2 = nplike([[[10]], [[10]]])
        inp3 = nplike([[[100]], [[100]]])
        inp4 = nplike([[[1000]], [[1000]]])
        out1 = a_norm(inp1, (2, 1, 1))
        out2 = a_norm(inp2, (2, 1, 1))
        out3 = a_norm(inp3, (2, 1, 1))
        out4 = a_norm(inp4, (2, 1, 1))

        res1 = out1.eval()
        res2 = out2.eval()
        res3 = out3.eval()
        res4 = out4.eval()
        v1 = 0.99997
        v2 = 9.9701046
        v3 = 77.6969504
        v4 = 61.7180374
        array_almost_equal(res1, A([[[v1]], [[v1]]]))
        array_almost_equal(res2, A([[[v2]], [[v2]]]))
        array_almost_equal(res3, A([[[v3]], [[v3]]]))
        array_almost_equal(res4, A([[[v4]], [[v4]]]))

    def test_case3_interval(self):
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        out = a_norm(iinp, (2, 1, 1))

        inp1 = A([[[1]], [[1]]])
        inp2 = A([[[10]], [[10]]])
        inp3 = A([[[100]], [[100]]])
        inp4 = A([[[1000]], [[1000]]])
        d1 = {tinpl: inp1, tinpu: inp1}
        d2 = {tinpl: inp2, tinpu: inp2}
        d3 = {tinpl: inp3, tinpu: inp3}
        d4 = {tinpl: inp4, tinpu: inp4}
        l1, u1 = out.eval(d1)
        l2, u2 = out.eval(d2)
        l3, u3 = out.eval(d3)
        l4, u4 = out.eval(d4)
        array_almost_equal(l1, u1)
        array_almost_equal(l2, u2)
        array_almost_equal(l3, u3)
        array_almost_equal(l4, u4)
        v1 = 0.99997
        v2 = 9.9701046
        v3 = 77.6969504
        v4 = 61.7180374
        array_almost_equal(l1, A([[[v1]], [[v1]]]))
        array_almost_equal(l2, A([[[v2]], [[v2]]]))
        array_almost_equal(l3, A([[[v3]], [[v3]]]))
        array_almost_equal(l4, A([[[v4]], [[v4]]]))

    def test_case4_interval(self):
        shp = (100, 1, 1)
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        out = a_norm(iinp, shp)

        inp = np.zeros(shp, dtype=theano.config.floatX)
        d = {tinpl: inp, tinpu: inp}
        inp[47, 0, 0] = inp[53, 0, 0] = 65536.0
        inp[48, 0, 0] = inp[49, 0, 0] = inp[50, 0, 0] = inp[51, 0, 0] = \
            inp[52, 0, 0] = 10.0
        l, u = out.eval(d)
        assert_almost_equal(l[50, 0, 0], u[50, 0, 0])
        assert_almost_equal(l[50, 0, 0], 9.9256, places=2)
        inp[48, 0, 0] = inp[49, 0, 0] = inp[50, 0, 0] = inp[51, 0, 0] = \
            inp[52, 0, 0] = 100.0
        l, u = out.eval(d)
        assert_almost_equal(l[50, 0, 0], u[50, 0, 0])
        assert_almost_equal(l[50, 0, 0], 59.4603, places=2)
        inp[48, 0, 0] = inp[49, 0, 0] = inp[50, 0, 0] = inp[51, 0, 0] = \
            inp[52, 0, 0] = 1000.0
        l, u = out.eval(d)
        assert_almost_equal(l[50, 0, 0], u[50, 0, 0])
        assert_almost_equal(l[50, 0, 0], 31.3876, places=2)
        assert_almost_equal(l[49, 0, 0], 0.1991, places=2)
        assert_almost_equal(l[51, 0, 0], 0.1991, places=2)

    def test_bitonicity_and_extremas_interval(self):
        shp = (5, 1, 1)
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        out = a_norm(iinp, shp)
        b = 200.0
        a = (2.0 * (50000.0 + b * b)) ** 0.5
        inp = A([[[b]], [[0.0]], [[a]], [[0.0]], [[0.0]]])
        d = {tinpl: inp, tinpu: inp}
        l1, _ = out.eval(d)
        inp[2, 0, 0] = a - 20.0
        l2, _ = out.eval(d)
        inp[2, 0, 0] = a + 20.0
        l3, _ = out.eval(d)
        inp[2, 0, 0] = a - 19.0
        l4, _ = out.eval(d)
        inp[2, 0, 0] = a - 18.0
        l5, _ = out.eval(d)
        inpl = inp.copy()
        inpu = inp.copy()
        d2 = {tinpl: inpl, tinpu: inpu}
        inpl[2, 0, 0] = a - 18.0
        inpu[2, 0, 0] = a + 20.0
        l6, _ = out.eval(d2)
        inpl[2, 0, 0] = a - 19.0
        inpu[2, 0, 0] = a + 20.0
        l7, _ = out.eval(d2)
        inpl[2, 0, 0] = a - 20.0
        inpu[2, 0, 0] = a + 20.0
        l8, _ = out.eval(d2)
        inpl[2, 0, 0] = a - 19.0
        inpu[2, 0, 0] = a + 20.0
        assert_greater(l1[2, 0, 0], l2[2, 0, 0])
        assert_greater(l1[2, 0, 0], l3[2, 0, 0])
        assert_greater(l5[2, 0, 0], l3[2, 0, 0])
        assert_almost_equal(l3[2, 0, 0], l6[2, 0, 0], places=2)
        assert_almost_equal(l3[2, 0, 0], l7[2, 0, 0], places=2)


class DropoutActivationTest(ActivationTest):

    def test_2x2_matrix(self):
        s, v, m = self.prepare()
        a = nplike([[[s(), s()], [s(), s()]]])
        res = a_dropout(a, 0.8)
        array_almost_equal(res.eval(), a.eval() * A([0.2]))


    def test_2x2_matrix_interval(self):
        s, v, m = self.prepare()
        l = A([[[s(), s()], [s(), s()]]])
        u = A([[[s(), s()], [s(), s()]]])
        tl, tu = T.dtensor3s('l', 'u')
        i = TheanoInterval(tl, tu)
        drp = a_dropout(i, 0.8)

        d = {tl: l, tu: u}
        (rl, ru) = drp.eval(d)
        array_almost_equal(rl, 0.2 * l)
        array_almost_equal(ru, 0.2 * u)


class ReluActivationTest(ActivationTest):

    def test_simple(self):
        inp = nplike([[[-3, -1, 1]]])
        array_almost_equal(a_relu(inp).eval(), A([[[0, 0, 1]]]))


    def test_interval_simple(self):
        inpl = A([[[-3, -1, 1]]])
        inpu = A([[[-2, 3, 2]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        res = a_relu(iinp)

        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        array_almost_equal(rl, A([[[0, 0, 1]]]))
        array_almost_equal(ru, A([[[0, 3, 2]]]))

    def test_interval_3D(self):
        inpl = A([[[-1, 2, -1], [0, 3, 5], [1, 2, 3]],
                  [[2, 3, 4], [-2, -3, -4], [-4, 0, 4]]])
        inpu = A([[[2, 2, 2], [1, 3, 5], [6, 5, 4]],
                  [[2, 3, 4], [-1, 0, 1], [4, 0, 4]]])
        tinpl, tinpu = T.dtensor3s('tinpl', 'tinpu')
        iinp = TheanoInterval(tinpl, tinpu)
        res = a_relu(iinp)

        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        array_almost_equal(rl, A([[[0, 2, 0], [0, 3, 5], [1, 2, 3]],
                                  [[2, 3, 4], [0, 0, 0], [0, 0, 4]]]))
        array_almost_equal(ru, A([[[2, 2, 2], [1, 3, 5], [6, 5, 4]],
                                  [[2, 3, 4], [0, 0, 1], [4, 0, 4]]]))

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
