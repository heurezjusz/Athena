"""Testing athenet.algorithm.derest.derivative functions.
"""

import numpy as np
import theano
import theano.tensor as T
import unittest
from math import e
from nose.tools import assert_almost_equal as aae, \
    assert_greater as ag
from numpy.testing import assert_array_almost_equal as arae
from athenet.algorithm.numlike import Interval as Itv, Nplike

from athenet.algorithm.derest.layers.fully_connected import d_fully_connected
from athenet.algorithm.derest.layers.convolutional import d_conv
from athenet.algorithm.derest.layers.pool import d_pool
from athenet.algorithm.derest.layers.dropout import d_dropout
from athenet.algorithm.derest.layers.norm import d_norm
from athenet.algorithm.derest.layers.relu import d_relu
from athenet.algorithm.derest.layers.softmax import d_softmax

theano.config.exception_verbosity = 'high'

A = np.array


def npl(x):
    return Nplike(A(x))


def thv(x):
    return theano.shared(np.array(x, dtype=float))


def ithv(x):
    v = thv(x)
    return Itv(v, v)


class DerivativeTest2(unittest.TestCase):

    def prepare(self):
        self.v = np.arange(24) + 3.0
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


class DerivativeTest(unittest.TestCase):
    pass


class FullyConnectedDerivativeTest(DerivativeTest):

    def test_1D_simple(self):
        dout = thv([[1]])
        idout = Itv(dout, dout)
        w = thv([[2]])
        shp = (1, 1)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[2]]))

    def test_2D_simple_used_1D_of_weights(self):
        dout = thv([[3, 6]])
        idout = Itv(dout, dout)
        w = thv([9, 12])
        shp = (1, 1)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[99]]))

    def test_2D_simple_used_2D_of_weights(self):
        dout = thv([[3, 0]])
        idout = Itv(dout, dout)
        w = thv([[6, 0], [9, 0]])
        shp = (1, 2)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[18, 27]]))

    def test_2D_1(self):
        dout = thv([[3, 6]])
        idout = Itv(dout, dout)
        w = thv([[9, 15], [12, 18]])
        shp = (1, 2)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[117, 144]]))

    def test_2D_Intervals(self):
        doutl = thv([[-3, -6, 3]])
        doutu = thv([[9, -3, 6]])
        idout = Itv(doutl, doutu)
        w = thv([[2, -3, -3], [-3, 1, 2], [5, -4, 3], [-2, -3, -4]])
        shp = (1, 2, 2)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, A([[[-15, -27], [6, -33]]]))
        arae(u, A([[[27, 18], [87, 12]]]))

    def test_2D_batches(self):
        dout = thv([[3, 6], [1, 2]])
        idout = Itv(dout, dout)
        w = thv([[9, 15], [12, 18]])
        shp = (2, 2)
        din = d_fully_connected(idout, w, shp)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[117, 144], [39, 48]]))


class ConvolutionalDerivativeTest(DerivativeTest):

    def test_dims(self):
        dout = ithv(np.ones((1, 2, 2, 4)))
        w = thv(np.ones((2, 1, 3, 4)))
        w = w[:, :, ::-1, ::-1]
        din = d_conv(dout, (1, 1, 4, 7), (2, 3, 4), w)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[[[2, 4, 6, 8, 6, 4, 2],
                     [4, 8, 12, 16, 12, 8, 4],
                     [4, 8, 12, 16, 12, 8, 4],
                     [2, 4, 6, 8, 6, 4, 2]]]]))

    def test_case(self):
        dout = ithv(A([[[[4, 8], [2, 3]]]]))
        w = thv(A([[[[2, 3, 0], [5, 7, 0], [0, 0, 0]]]]))
        w = w[:, :, ::-1, ::-1]
        din = d_conv(dout, (1, 1, 2, 2), (1, 3, 3), w, padding=(1, 1))
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[[[80, 65], [29, 21]]]]))

    def test_all_dims(self):
        dout = ithv(A([[[[2, 3], [5, 7]], [[0.2, 0.3], [0.5, 0.7]]]]))
        w = thv(A([[[[1, 0, 2], [0, 4, 0], [3, 0, 0]],
                    [[0, 0, 0], [0, 9, 10], [0, 11, 12]]],
                   [[[5, 0, 6], [0, 0, 0], [7, 0, 8]],
                    [[13, 15, 0], [0, 0, 0], [14, 16, 0]]]]))
        w = w[:, :, ::-1, ::-1]
        din = d_conv(dout, (1, 2, 2, 2), (2, 3, 3), w, padding=(1, 1))
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[[[18.5, 25], [31.1, 29.6]],
                    [[34.6, 57.5], [74.4, 174.8]]]]))

    def test_interval(self):
        # TODO: this
        doutl = thv(A([[[[-2, -3], [-4, 5]]]]))
        doutu = thv(A([[[[-1, -2], [1, 6]]]]))
        dout = Itv(doutl, doutu)
        w = thv([[[[-1, 1], [-2, 3]]]])
        din = d_conv(dout, (1, 1, 3, 3), (1, 2, 2), w)
        l, u = din.eval()


class MaxPoolDerivativeTest(DerivativeTest):

    def test_simple(self):
        inpl = thv([[[[1, 1], [1, 1]]]])
        inpu = thv([[[[2, 2], [2, 2]]]])
        iinp = Itv(inpl, inpu)
        idout = ithv([[[[5]]]])
        shp = (1, 1, 2, 2)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='max')
        l, u = din.eval()
        arae(l, A([[[[0, 0], [0, 0]]]]))
        arae(u, A([[[[5, 5], [5, 5]]]]))

    def test_neg_output(self):
        inpl = thv([[[[1, 1], [1, 1]]]])
        inpu = thv([[[[2, 2], [2, 2]]]])
        iinp = Itv(inpl, inpu)
        idout = ithv([[[[-3]]]])
        shp = (1, 1, 2, 2)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='max')
        l, u = din.eval()
        arae(l, A([[[[-3, -3], [-3, -3]]]]))
        arae(u, A([[[[0, 0], [0, 0]]]]))

    def test_2D(self):
        inpl = thv([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inpu = thv([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[[[-1, -2], [-3, -4]]]])
        doutu = thv([[[[5, 4], [3, 2]]]])
        idout = Itv(doutl, doutu)
        shp = (1, 1, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='max')
        l, u = din.eval()
        arae(l, A([[[[-1, -3, -2], [-4, -10, -6], [-3, -7, -4]]]]))
        arae(u, A([[[[5, 9, 4], [8, 14, 6], [3, 5, 2]]]]))

    def test_channels_batch(self):
        inpl = thv([[
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [3, 0, 0]]
                     ],
                    [
                     [[0, 3, 3], [4, 5, 6], [7, 8, 4]],
                     [[-3, -3, -3], [-3, -3, -3], [3, 3, 3]]
                    ]])
        inpu = thv([[
                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                     [[1, 1, 1], [1, 1, 1], [4, 1, 1]]
                    ],
                    [
                     [[2, 4, 4], [9, 9, 9], [9, 9, 9]],
                     [[2, 2, 2], [2, 2, 2], [5, 5, 5]]
                    ]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[
                      [[-1, -2], [-3, -4]],
                      [[1, 2], [-3, -2]]
                     ],
                     [
                      [[1, 2], [-3, -2]],
                      [[-1, 1], [-1, 1]]
                     ]])
        doutu = thv([[
                      [[5, 4], [3, 2]],
                      [[4, 4], [4, 4]],
                     ],
                     [
                      [[4, 5], [0, 1]],
                      [[0, 2], [0, 2]]
                     ]])
        idout = Itv(doutl, doutu)
        shp = (2, 2, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='max')
        l, u = din.eval()
        arae(l, A([[
                    [[-1, -3, -2], [-4, -10, -6], [-3, -7, -4]],
                    [[0, 0, 0], [0, -2, -2], [-3, -2, -2]]
                   ],
                   [
                    [[0, 0, 0], [-3, -5, -2], [-3, -5, -2]],
                    [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]
                   ]]))
        arae(u, A([[
                    [[5, 9, 4], [8, 14, 6], [3, 5, 2]],
                    [[4, 8, 4], [4, 12, 8], [4, 4, 4]]
                   ],
                   [
                    [[0, 0, 0], [4, 10, 6], [0, 1, 1]],
                    [[0, 2, 2], [0, 2, 2], [0, 2, 2]]
                   ]]))

    def test_stride(self):
        tinpl = theano.shared(np.arange(25).reshape((1, 1, 5, 5)))
        tinpu = theano.shared(np.arange(25).reshape((1, 1, 5, 5)) + 2)
        iinp = Itv(tinpl, tinpu)
        idout = ithv([[[[-1, 2], [-3, 4]]]])
        shp = (1, 1, 5, 5)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), stride=(3, 3),
                     mode='max')
        l, u = din.eval()
        arae(l, A([[[[0, 0, 0, 0, 0],
                     [-1, -1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [-3, -3, 0, 0, 0]]]]))
        arae(u, A([[[[0, 0, 0, 0, 0],
                     [0, 0, 0, 2, 2],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4]]]]))

    def test_padding(self):
        inpl = thv([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inpu = thv([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[[[-1, -2], [-3, -4]]]])
        doutu = thv([[[[5, 4], [3, 2]]]])
        idout = Itv(doutl, doutu)
        shp = (1, 1, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), padding=(1, 1),
                     stride=(3, 3), mode='max')
        l, u = din.eval()
        arae(l, A([[[[-1, 0, -2], [0, 0, 0], [-3, 0, -4]]]]))
        arae(u, A([[[[5, 0, 4], [0, 0, 0], [3, 0, 2]]]]))


class AvgPoolDerivativeTest(DerivativeTest):

    def test_simple(self):
        inpl = thv([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inpu = thv([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[[[-1, -2], [-3, -4]]]])
        doutu = thv([[[[5, 4], [3, 2]]]])
        idout = Itv(doutl, doutu)
        shp = (1, 1, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='avg')
        l, u = din.eval()
        arae(l, A([[[[-1, -3, -2], [-4, -10, -6], [-3, -7, -4]]]]) / 4.0)
        arae(u, A([[[[5, 9, 4], [8, 14, 6], [3, 5, 2]]]]) / 4.0)

    def test_channels_batch(self):
        inpl = thv([[
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [3, 0, 0]]
                    ],
                    [
                     [[0, 3, 3], [4, 5, 6], [7, 8, 4]],
                     [[-3, -3, -3], [-3, -3, -3], [3, 3, 3]]
                    ]])
        inpu = thv([[
                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                     [[1, 1, 1], [1, 1, 1], [4, 1, 1]]
                    ],
                    [
                     [[2, 4, 4], [9, 9, 9], [9, 9, 9]],
                     [[2, 2, 2], [2, 2, 2], [5, 5, 5]]
                    ]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[
                      [[-1, -2], [-3, -4]],
                      [[1, 2], [-3, -2]]
                     ],
                     [
                      [[1, 2], [-3, -2]],
                      [[-1, 1], [-1, 1]]
                     ]])
        doutu = thv([[
                      [[5, 4], [3, 2]],
                      [[4, 4], [4, 4]],
                     ],
                     [
                      [[4, 5], [0, 1]],
                      [[0, 2], [0, 2]]
                     ]])
        idout = Itv(doutl, doutu)
        shp = (2, 2, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), mode='avg')
        l, u = din.eval()
        arae(l, A([[
                    [[-1, -3, -2], [-4, -10, -6], [-3, -7, -4]],
                    [[1, 3, 2], [-2, -2, 0], [-3, -5, -2]]
                   ],
                   [
                    [[1, 3, 2], [-2, -2, 0], [-3, -5, -2]],
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                   ]]) / 4.0)
        arae(u, A([[
                    [[5, 9, 4], [8, 14, 6], [3, 5, 2]],
                    [[4, 8, 4], [8, 16, 8], [4, 8, 4]]
                   ],
                   [
                    [[4, 9, 5], [4, 10, 6], [0, 1, 1]],
                    [[0, 2, 2], [0, 4, 4], [0, 2, 2]]
                   ]]) / 4.0)

    def test_stride(self):
        tinpl = theano.shared(np.arange(25).reshape((1, 1, 5, 5)))
        tinpu = theano.shared(np.arange(25).reshape((1, 1, 5, 5)) + 2)
        iinp = Itv(tinpl, tinpu)
        idout = ithv([[[[-1, 2], [-3, 4]]]])
        shp = (1, 1, 5, 5)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), stride=(3, 3),
                     mode='avg')
        l, u = din.eval()
        arae(l, A([[[[-1, -1, 0, 2, 2],
                     [-1, -1, 0, 2, 2],
                     [0, 0, 0, 0, 0],
                     [-3, -3, 0, 4, 4],
                     [-3, -3, 0, 4, 4]]]]) / 4.0)
        arae(u, A([[[[-1, -1, 0, 2, 2],
                     [-1, -1, 0, 2, 2],
                     [0, 0, 0, 0, 0],
                     [-3, -3, 0, 4, 4],
                     [-3, -3, 0, 4, 4]]]]) / 4.0)

    def test_padding(self):
        inpl = thv([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inpu = thv([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        iinp = Itv(inpl, inpu)
        doutl = thv([[[[-1, -2], [-3, -4]]]])
        doutu = thv([[[[5, 4], [3, 2]]]])
        idout = Itv(doutl, doutu)
        shp = (1, 1, 3, 3)
        din = d_pool(idout, iinp, shp, poolsize=(2, 2), padding=(1, 1),
                     stride=(3, 3), mode='avg')
        l, u = din.eval()
        arae(l, A([[[[-1, 0, -2], [0, 0, 0], [-3, 0, -4]]]]) / 4.0)
        arae(u, A([[[[5, 0, 4], [0, 0, 0], [3, 0, 2]]]]) / 4.0)


class SoftmaxDerivativeTest(DerivativeTest):

    def test_case1(self):
        dout = Itv.derest_output(1)
        din = d_softmax(dout)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[1]]))

    def test_case2(self):
        dout = Itv.derest_output(3)
        din = d_softmax(dout)
        l, u = din.eval()
        arae(l, u)
        arae(l, A([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


class NormDerivativeTest(DerivativeTest2):

    alpha = 0.00002
    beta = 0.75
    k = 1.0
    n = 5


class DropoutDerivativeTest(DerivativeTest):

    def test_case1(self):
        doutl = thv([[-3, 0, 3], [3, -3, -5], [-3, -2, 1]])
        doutu = thv([[-3, 2, 3], [5, 3, 2], [-1, 3, 3]])
        idout = Itv(doutl, doutu)
        idin = d_dropout(idout, 0.8)
        l, u = idin.eval()
        rl = A([[-0.6, 0, 0.6], [0.6, -0.6, -1], [-0.6, -0.4, 0.2]])
        ru = A([[-0.6, 0.4, 0.6], [1, 0.6, 0.4], [-0.2, 0.6, 0.6]])
        arae(l, rl)
        arae(u, ru)


class ReluDerivativeTest(DerivativeTest):

    def test_case1(self):
        shp = (4, 3, 2)
        act = np.zeros(shp)
        for n_in in range(4):
            for h in range(3):
                for w in range(2):
                    act[n_in, h, w] += 100 * n_in + 10 * h + w
        thact = theano.shared(act)
        iact = Itv(thact, thact)
        thdout = theano.shared(np.ones(shp))
        idout = Itv(thdout, thdout)
        idin = d_relu(idout, iact)
        l, u = idin.eval()
        aae(l[0, 0, 0], 0.0)
        aae(u[0, 0, 0], 1.0)
        aae(l[2, 1, 1], 1.0)
        aae(l[2, 2, 1], 1.0)
        aae(l[1, 0, 1], 1.0)
        aae(l[2, 1, 1], 1.0)
        aae(l[2, 2, 0], 1.0)
        aae(l[1, 0, 1], 1.0)

    def test_case2(self):
        actl = thv([-2, -1, -1, 0, 0, 1])
        actu = thv([-1, 1, 0, 0, 1, 2])
        doutl = thv([2, 3, 4, 7, 11, 13])
        doutu = thv([3, 5, 7, 11, 13, 17])
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(idout, iact)
        l, u = idin.eval()
        rl = A([0, 0, 0, 0, 0, 13])
        ru = A([0, 5, 7, 11, 13, 17])
        arae(l, rl)
        arae(u, ru)

    def test_case3(self):
        actl = thv([-2, -1, -1, 0, 0, 1])
        actu = thv([-1, 1, 0, 0, 1, 2])
        doutl = thv([-3, -5, -7, -11, -13, -17])
        doutu = thv([-2, -3, -5, -7, -11, -13])
        iact = Itv(actl, actu)
        idout = Itv(doutl, doutu)
        idin = d_relu(idout, iact)
        l, u = idin.eval()
        rl = A([0, -5, -7, -11, -13, -17])
        ru = A([0, 0, 0, 0, 0, -13])
        arae(l, rl)
        arae(u, ru)

if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
