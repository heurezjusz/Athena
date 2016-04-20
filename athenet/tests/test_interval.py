"""Testing athenet.algorithm.numlike.Interval class with its methods.
"""

import unittest
from nose.tools import assert_is, assert_equal
from athenet.algorithm.numlike.interval import Interval, \
    NEUTRAL_INTERVAL_LOWER, NEUTRAL_INTERVAL_UPPER, \
    DEFAULT_INTERVAL_LOWER, DEFAULT_INTERVAL_UPPER
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal as arae
import theano
import theano.tensor as T
from theano import function


def A(x):
    return np.array(x, dtype=theano.config.floatX)


class IntervalTest(unittest.TestCase):

    def test_interval_scalar(self):
        x, y = T.scalars('x', 'y')
        i = Interval(x, y)
        l, u = i.lower, i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u - l
        f = function([x, y], z)
        arae(f(1.1, 3.2), 2.1)

    def test_interval_matrix(self):
        x, y = T.matrices('x', 'y')
        i = Interval(x, y)
        l, u = i.lower, i.upper
        assert_is(x, l)
        assert_is(y, u)
        z = u + l
        f = function([x, y], z)
        res_z = f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
        res = A([[11., 22.], [33., 44.]])
        assert_array_equal(res_z, res)

    def test_getitem(self):
        x, y = T.matrices('x', 'y')
        i = Interval(x, y)
        i0, i1, i2 = i[0, 0], i[1, 1], i[2, 2]
        l0, l1, l2 = i0.lower, i1.lower, i2.lower
        u0, u1, u2 = i0.upper, i1.upper, i2.upper
        f = function([x, y], [l0, l1, l2, u0, u1, u2])
        ex_x = A([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ex_y = ex_x * 10
        [rl0, rl1, rl2, ru0, ru1, ru2] = f(ex_x, ex_y)
        assert_equal(rl0, 1)
        assert_equal(rl1, 5)
        assert_equal(rl2, 9)
        assert_equal(ru0, 10)
        assert_equal(ru1, 50)
        assert_equal(ru2, 90)

    def test_setitem(self):
        x, y, z, w = T.matrices('x', 'y', 'z', 'w')
        i1 = Interval(x, y)
        i2 = Interval(z, w)
        i1[:, 1:3] = i2
        f = function([x, y, z, w], [i1.lower, i1.upper])
        ex_x = A([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ex_z = A([[20, 30], [50, 60], [80, 90]])
        ex_y = 100 * ex_x
        ex_w = 100 * ex_z
        l, u = f(ex_x, ex_y, ex_z, ex_w)
        rl = A([[1., 20., 30.], [4., 50., 60.], [7., 80., 90.]])
        ru = rl * 100
        assert_array_equal(l, rl)
        assert_array_equal(u, ru)

    def test_shape(self):
        x, y = T.matrices('x', 'y')
        i1 = Interval(x, y)
        ex_x = A([[2, 3], [5, 6], [8, 9]])
        shp = i1.shape
        rshp = shp.eval({x: ex_x})
        assert_equal(len(rshp), 2)
        assert_equal(rshp[0], 3)
        assert_equal(rshp[1], 2)

    def test_ops1(self):
        # __add__, __sub__, __mul__,
        x, y, z, w = T.matrices('x', 'y', 'z', 'w')
        i1 = Interval(x, y)
        i2 = Interval(z, w)
        l1 = A([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        u1 = l1 * 10 + 1
        l2 = l1 * 10 + 2
        u2 = l1 * 10 + 3
        r_add1 = i1 + i2
        r_add2 = i1 + x
        r_sub1 = i1 - i2
        r_sub2 = i1 - x
        r_mul1 = i1 * i2
        r_mul2 = i1 * x
        fres = [r_add1.lower, r_add1.upper, r_add2.lower, r_add2.upper,
                r_sub1.lower, r_sub1.upper, r_sub2.lower, r_sub2.upper,
                r_mul1.lower, r_mul1.upper, r_mul2.lower, r_mul2.upper]
        f = function([x, y, z, w], fres)
        add1l, add1u, add2l, add2u, \
            sub1l, sub1u, sub2l, sub2u, \
            mul1l, mul1u, mul2l, mul2u, = f(l1, u1, l2, u2)
        ops_results = [add1l, add1u, add2l, add2u,
                       sub1l, sub1u, sub2l, sub2u,
                       mul1l, mul1u, mul2l, mul2u]
        r_add1l = l1 + l2
        r_add2l = l1 + l1
        r_add1u = u1 + u2
        r_add2u = u1 + l1
        r_sub1l = l1 - u2
        r_sub2l = l1 - l1
        r_sub1u = u1 - l2
        r_sub2u = u1 - l1
        r_mul1l = l1 * l2
        r_mul2l = l1 * l1
        r_mul1u = u1 * u2
        r_mul2u = u1 * l1
        results = [r_add1l, r_add1u, r_add2l, r_add2u,
                   r_sub1l, r_sub1u, r_sub2l, r_sub2u,
                   r_mul1l, r_mul1u, r_mul2l, r_mul2u]
        for i in range(len(ops_results)):
            arae(ops_results, results)

    def test_ops2(self):
        # reciprocal, neg, exp, square
        x, y = T.matrices('x', 'y')
        i = Interval(x, y)
        reciprocal = i.reciprocal()
        neg = i.neg()
        exp = i.exp()
        sq = i.square()
        recl, recu = reciprocal.lower, reciprocal.upper
        negl, negu = neg.lower, neg.upper
        expl, expu = exp.lower, exp.upper
        sql, squ = sq.lower, sq.upper
        l = A([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        u = l + 10
        f = function([x, y], [recl, recu, negl, negu, expl, expu, sql, squ])
        rrecl, rrecu, rnegl, rnegu, rexpl, rexpu, rsql, rsqu = f(l, u)
        threcl = np.reciprocal(u)
        threcu = np.reciprocal(l)
        thnegl = -u
        thnegu = -l
        thexpl = np.exp(l)
        thexpu = np.exp(u)
        thsql = np.square(l)
        thsqu = np.square(u)
        assert_array_equal(rrecl, threcl)
        assert_array_equal(rrecu, threcu)
        assert_array_equal(rnegl, thnegl)
        assert_array_equal(rnegu, thnegu)
        assert_array_equal(rexpl, thexpl)
        assert_array_equal(rexpu, thexpu)
        assert_array_equal(rsql, thsql)
        assert_array_equal(rsqu, thsqu)

    def test_ops3(self):
        # __div__, __rdiv__
        l1, l2, u1, u2 = T.vectors('l1', 'l2', 'u1', 'u2')
        i1 = Interval(l1, u1)
        i2 = Interval(l2, u2)
        r1 = i1 / i2
        v1l = A([3.0, -4.0, 3.0, -4.0, -4.0, -4.0])
        v1u = A([4.0, -3.0, 4.0, -3.0, 3.0, 3.0])
        v2l = A([5.0, 5.0, -6.0, -6.0, 5.0, -6.0])
        v2u = A([6.0, 6.0, -5.0, -5.0, 6.0, -5.0])
        d12 = {l1: v1l, l2: v2l, u1: v1u, u2: v2u}
        res1 = r1.eval(d12)
        ll = 3.0 / 5.0
        lu = 3.0 / 6.0
        ul = 4.0 / 5.0
        ans1l = A([lu, -ul, -ul, lu, -ul, -ll])
        ans1u = A([ul, -lu, -lu, ul, ll, ul])
        vl = A([-4.0, -4.0, 3.0])
        vu = A([-3.0, 3.0, 4.0])
        v = 7.0
        d1 = {l1: vl, u1: vu}
        r2 = i1 / v
        res2 = r2.eval(d1)
        l = 3.0 / 7.0
        u = 4.0 / 7.0
        ans2l = A([-u, -u, l])
        ans2u = A([-l, l, u])
        vl = A([-4.0, 3.0])
        vu = A([-3.0, 4.0])
        d1 = {l1: vl, u1: vu}
        r3 = v / i1
        res3 = r3.eval(d1)
        l = 7.0 / 3.0
        u = 7.0 / 4.0
        ans3l = A([-l, u])
        ans3u = A([-u, l])
        arae(res1[0], ans1l)
        arae(res1[1], ans1u)
        arae(res2[0], ans2l)
        arae(res2[1], ans2u)
        arae(res3[0], ans3l)
        arae(res3[1], ans3u)

    def test_ops4(self):
        # power
        x, y = T.vectors('x', 'y')
        itv = Interval(x, y)
        v1l = A([-3, -2, -1, -2, 0.5, 0.5, 1, 2])
        v1u = A([-2, -1, -0.5, -0.5, 2, 1, 2, 3])
        v2l = A([1, 2])
        v2u = A([3, 4])
        v3l = A([-2., -2., -2., -1., -1., -1., -0.5, -0.5, -0.5])
        v3u = A([0.5, 1., 2., 0.5, 1., 2., 0.5, 1., 2.])
        v1 = (v1l, v1u)
        v2 = (v2l, v2u)
        v3 = (v3l, v3u)
        exponents1 = [-3, -2, 2, 3]
        exponents2 = [-2.5, -2., 2., 2.5]
        exponents3 = [2, 3]

        def make_power(exp):
            return itv.power(exp)

        powers1 = map(make_power, exponents1)
        powers2 = map(make_power, exponents2)
        powers3 = map(make_power, exponents3)

        def make_lu(power):
            return power.lower, power.upper

        lus1 = map(make_lu, powers1)
        lus2 = map(make_lu, powers2)
        lus3 = map(make_lu, powers3)

        def make_function((l, u)):
            return function([x, y], [l, u])

        functions1 = map(make_function, lus1)
        functions2 = map(make_function, lus2)
        functions3 = map(make_function, lus3)

        def make_res1(f):
            return f(*v1)

        def make_res2(f):
            return f(*v2)

        def make_res3(f):
            return f(*v3)

        res1 = map(make_res1, functions1)
        res2 = map(make_res2, functions2)
        res3 = map(make_res3, functions3)
        ans1l = [A([4., 1., 0.25, 0.25, 0.25, 0.25, 1., 4.]),
                 A([-27., -8., -1., -8., 0.125, 0.125, 1., 8.])]
        ans1u = [A([9., 4., 1., 4., 4., 1., 4., 9.]),
                 A([-8., -1., -0.125, -0.125, 8., 1., 8., 27.])]
        ans1l = [np.reciprocal(ans1u[1]), np.reciprocal(ans1u[0])] + ans1l
        ans1u = [np.reciprocal(ans1l[3]), np.reciprocal(ans1l[2])] + ans1u
        ans2l = [A([1., 4.]), A([1., 2. ** 2.5])]
        ans2u = [A([9., 16.]), A([3. ** 2.5, 4. ** 2.5])]
        ans2l = [np.reciprocal(ans2u[1])] + [np.reciprocal(ans2u[0])] + ans2l
        ans2u = [np.reciprocal(ans2l[3])] + [np.reciprocal(ans2l[2])] + ans2u
        ans3l = [A([0.] * 9),
                 A([-8., -8., -8., -1., -1., -1., -0.125, -0.125, -0.125])]
        ans3u = [A([4., 4., 4., 1., 1., 4., 0.25, 1., 4.]),
                 A([0.125, 1., 8., 0.125, 1., 8., 0.125, 1., 8.])]
        for i in range(4):
            arae(res1[i][0], ans1l[i])
            arae(res1[i][1], ans1u[i])
            arae(res2[i][0], ans2l[i])
            arae(res2[i][1], ans2u[i])
        for i in range(2):
            arae(res3[i][0], ans3l[i])
            arae(res3[i][1], ans3u[i])

    def test_dot(self):
        inpl = A([[[0, 1]]])
        inpu = A([[[2, 3]]])
        w = A([[4, -5, 6], [7, 8, 9]])
        b = A([1, 3, 5])
        crl = A([0 * 4 + 1 * 7 + 1,
                 2 * (-5) + 1 * 8 + 3,
                 0 * 6 + 1 * 9 + 5])
        cru = A([2 * 4 + 3 * 7 + 1,
                 0 * (-5) + 3 * 8 + 3,
                 2 * 6 + 3 * 9 + 5])
        tinpl, tinpu = T.tensor3s('inpl', 'inpu')
        iinp = Interval(tinpl, tinpu)
        res = iinp.flatten().dot(w)
        res += b
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, crl)
        arae(ru, cru)

    def test_max(self):
        al = A([[1, 2], [3, 4]])
        au = A([[2, 2], [4, 7]])
        bl = A([[0, 3], [3, -4]])
        bu = A([[2, 4], [3, -3]])
        alt, aut, blt, but = T.matrices('alt', 'aut', 'blt', 'but')
        ai = Interval(alt, aut)
        bi = Interval(blt, but)
        ci = ai.max(bi)
        d = {alt: al, aut: au, blt: bl, but: bu}
        res = ci.eval(d)
        rl = res[0]
        ru = res[1]
        ansl = A([[1, 3], [3, 4]])
        ansu = A([[2, 4], [4, 7]])
        arae(rl, ansl)
        arae(ru, ansu)

    def test_amax(self):
        al = A([[1, 2], [3, 4]])
        au = A([[2, 2], [4, 7]])
        alt, aut = T.matrices('alt', 'aut')
        ai = Interval(alt, aut)
        ci = ai.amax(axis=1, keepdims=True)
        d = {alt: al, aut: au}
        res = ci.eval(d)
        rl = res[0]
        ru = res[1]
        ansl = A([[2], [4]])
        ansu = A([[2], [7]])
        arae(rl, ansl)
        arae(ru, ansu)

    def test_reshape(self):
        tl, tu = T.matrices('l', 'u')
        xl = A([[1, 2, 3], [4, 5, 6]])
        xu = xl + 3
        i = Interval(tl, tu)
        i1 = i.reshape((1, 6))
        i2 = i.reshape((2, 3))
        i3 = i.reshape((3, 2))
        i4 = i.reshape((6, 1))
        [l1, u1] = i1.eval({tl: xl, tu: xu})
        [l2, u2] = i2.eval({tl: xl, tu: xu})
        [l3, u3] = i3.eval({tl: xl, tu: xu})
        [l4, u4] = i4.eval({tl: xl, tu: xu})
        assert_array_equal(l1, xl.reshape((1, 6)))
        assert_array_equal(l2, xl.reshape((2, 3)))
        assert_array_equal(l3, xl.reshape((3, 2)))
        assert_array_equal(l4, xl.reshape((6, 1)))
        assert_array_equal(u1, xu.reshape((1, 6)))
        assert_array_equal(u2, xu.reshape((2, 3)))
        assert_array_equal(u3, xu.reshape((3, 2)))
        assert_array_equal(u4, xu.reshape((6, 1)))

    def test_flatten(self):
        t1 = T.dvector('t1')
        t2 = T.matrix('t2')
        t3 = T.tensor3('t3')
        t4 = T.tensor4('t4')
        i1 = Interval(t1, t1)
        i2 = Interval(t2, t2)
        i3 = Interval(t3, t3)
        i4 = Interval(t4, t4)
        f1 = i1.flatten()
        f2 = i2.flatten()
        f3 = i3.flatten()
        f4 = i4.flatten()
        v1 = np.arange(36, dtype=theano.config.floatX)
        v2 = np.arange(36, dtype=theano.config.floatX)
        v3 = np.arange(36, dtype=theano.config.floatX)
        v4 = np.arange(36, dtype=theano.config.floatX)
        v1.resize((36,))
        v2.resize((4, 9))
        v3.resize((2, 2, 9))
        v4.resize((2, 3, 2, 3))
        [l1, u1] = f1.eval({t1: v1})
        [l2, u2] = f2.eval({t2: v2})
        [l3, u3] = f3.eval({t3: v3})
        [l4, u4] = f4.eval({t4: v4})
        assert_array_equal(v1, l1)
        assert_array_equal(v1, l2)
        assert_array_equal(v1, l3)
        assert_array_equal(v1, l4)
        assert_array_equal(v1, u1)
        assert_array_equal(v1, u2)
        assert_array_equal(v1, u3)
        assert_array_equal(v1, u4)

    def test_sum(self):
        vl = A([[[-3, 2],
                 [5, 6]],
                [[1, -1],
                 [9, 8]]])
        n = 10
        vu = A([[[n, n],
                 [n, n]],
                [[n, n],
                 [n, n]]])
        tvl, tvu = T.tensor3s('tvl', 'tvu')
        d = {tvl: vl, tvu: vu}
        itv = Interval(tvl, tvu)
        res1 = itv.sum(axis=0, keepdims=False)
        res2 = itv.sum(axis=1, keepdims=False)
        res3 = itv.sum(axis=2, keepdims=False)
        res4 = itv.sum(axis=0, keepdims=True)
        res5 = itv.sum(axis=1, keepdims=True)
        res6 = itv.sum(axis=2, keepdims=True)
        l1, _ = res1.eval(d)
        l2, _ = res2.eval(d)
        l3, _ = res3.eval(d)
        l4, _ = res4.eval(d)
        l5, _ = res5.eval(d)
        l6, _ = res6.eval(d)
        arae(l1, A([[-2, 1], [14, 14]]))
        arae(l2, A([[2, 8], [10, 7]]))
        arae(l3, A([[-1, 11], [0, 17]]))
        arae(l4, A([[[-2, 1], [14, 14]]]))
        arae(l5, A([[[2, 8]], [[10, 7]]]))
        arae(l6, A([[[-1], [11]], [[0], [17]]]))

    def test_abs(self):
        vl = A([[[-3, 2],
                 [5, 6]],
                [[1, -1],
                 [9, 8]]])
        vu = A([[[-2, 3],
                 [5, 7]],
                [[1, 1],
                 [9, 9]]])
        tvl, tvu = T.tensor3s('tvl', 'tvu')
        d = {tvl: vl, tvu: vu}
        itv = Interval(tvl, tvu)
        res = itv.abs()
        l, u = res.eval(d)
        arae(l, A([[[2, 2], [5, 6]],
                   [[1, 0], [9, 8]]]))
        arae(u, A([[[3, 3], [5, 7]],
                   [[1, 1], [9, 9]]]))

    def test_T(self):
        vl = A([[1, 2], [3, 4]])
        vu = A([[5, 6], [7, 8]])
        tvl, tvu = T.matrices('tvl', 'tvu')
        d = {tvl: vl, tvu: vu}
        itv = Interval(tvl, tvu)
        res = itv.T
        l, u = res.eval(d)
        arae(l, A([[1, 3], [2, 4]]))
        arae(u, A([[5, 7], [6, 8]]))

    def test_from_shape(self):
        shp = (3, 4)
        np_shp = A([3, 4])
        i = Interval.from_shape(shp, neutral=True)
        assert_array_equal(i.shape.eval(), np_shp)
        assert_array_equal(i.lower.shape.eval(), np_shp)
        assert_array_equal(i.upper.shape.eval(), np_shp)
        l, u = i.eval()
        arae(l, np.ones(shp, dtype=theano.config.floatX) *
             NEUTRAL_INTERVAL_LOWER)
        arae(u, np.ones(shp, dtype=theano.config.floatX) *
             NEUTRAL_INTERVAL_UPPER)
        i = Interval.from_shape(shp, neutral=False)
        assert_array_equal(i.shape.eval(), np_shp)
        assert_array_equal(i.lower.shape.eval(), np_shp)
        assert_array_equal(i.upper.shape.eval(), np_shp)
        l, u = i.eval()
        arae(l, np.ones(shp, dtype=theano.config.floatX) *
             DEFAULT_INTERVAL_LOWER)
        arae(u, np.ones(shp, dtype=theano.config.floatX) *
             DEFAULT_INTERVAL_UPPER)

    def test_eval(self):
        txl, txu, tyl, tyu = T.matrices('xl', 'xu', 'yl', 'yu')
        xl = A([[1, 2], [3, 4]])
        xu = A([[2, 4], [6, 9]])
        yl = A([[-1, -5], [0, 3]])
        yu = A([[4, 2], [0, 3]])
        ix = Interval(txl, txu)
        iy = Interval(tyl, tyu)
        iz = ix + iy
        d = {txl: xl, txu: xu, tyl: yl, tyu: yu}
        zl, zu = iz.eval(d)
        arae(zl, xl + yl)
        arae(zu, xu + yu)
        i2 = Interval(1, 3)
        i2l, i2u = i2.eval()
        assert_equal(i2l, 1)
        assert_equal(i2u, 3)
        i2l, i2u = i2.eval({})
        assert_equal(i2l, 1)
        assert_equal(i2u, 3)

    def test_op_relu(self):
        inpl = A([[[-3, -1, 1]]])
        inpu = A([[[-2, 3, 2]]])
        tinpl, tinpu = T.tensor3s('tinpl', 'tinpu')
        iinp = Interval(tinpl, tinpu)
        res = iinp.op_relu()
        d = {tinpl: inpl, tinpu: inpu}
        rl, ru = res.eval(d)
        arae(rl, A([[[0, 0, 1]]]))
        arae(ru, A([[[0, 3, 2]]]))

    def test_derest_output(self):
        o1 = Interval.derest_output(1)
        o4 = Interval.derest_output(4)
        l1, u1 = o1.eval()
        l4, u4 = o4.eval()
        arae(l1, u1)
        arae(l4, u4)
        arae(l1, A([[1]]))
        arae(l4, A([[1, 0, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1]]))


if __name__ == '__main__':
    unittest.main(verbosity=2, catchbreak=True)
