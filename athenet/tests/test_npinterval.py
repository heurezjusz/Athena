from athenet.algorithm.numlike import NpInterval
from unittest import TestCase, main, expectedFailure
from random import randrange, randint, uniform
from itertools import product
from numpy.testing import assert_array_almost_equal
import numpy as np


def _random_shape(n=None, limit=None):
    if n is None:
        n = randrange(1, 7)
    result = None
    if limit is None:
        limit = 10 ** 4
    size = 1
    for i in xrange(n):
        l = randrange(1, 10)
        if result is None:
            result = (l,)
        else:
            result += (l,)
        size *= l
        if size >= limit:
            break
    while len(result) < n:
        result += (1,)
    return result


def _random_npinterval(shape=None, dim=None):
    if shape is None:
        shape = _random_shape(dim)
    r1 = np.random.rand(*shape) * 20 - 10
    r2 = np.random.rand(*shape) * 20 - 10
    return NpInterval(np.minimum(r1, r2), np.maximum(r1, r2))


def _rand_from_npinterval(I):
    l = I.lower
    u = I.upper
    return (l + np.random.rand(*I.shape) * (u - l))


class TestLocalFunctions(TestCase):
    def test_rand_from(self):
        for _ in xrange(100):
            I = _random_npinterval()
            r = _rand_from_npinterval(I)
            self.assertTrue((I.lower <= r).all())
            self.assertTrue((r <= I.upper).all())


class TestNpInterval(TestCase):
    def _random_shape(self, size_limit=10**4, dimensions_limit=7):
        result = None
        size = 1
        for i in xrange(randrange(1, dimensions_limit)):
            l = randrange(1, 10)
            if result is None:
                result = (l,)
            else:
                result += (l,)
            size *= l
            if size >= size_limit:
                return result
        return result

    def _check_lower_upper(self, a):
        self.assertTrue((a.lower <= a.upper).all())

    def _random_npinterval(self, shape=None, size_limit=10**2,
                           number_limit=10**2):
        if shape is None:
            shape = self._random_shape(size_limit, 4)
        a = np.random.rand(*shape) * uniform(-number_limit, number_limit)
        b = np.random.rand(*shape) * uniform(-number_limit, number_limit)
        return NpInterval(np.minimum(a, b), np.maximum(a, b))

    def _random_ndarray(self, shape=None):
        if shape is None:
            shape = self._random_shape(10**2, 4)
        return np.random.rand(*shape)

    def _assert_npintervals_equal(self, a, b):
        self.assertTrue((a.lower == b.lower).all())
        self.assertTrue((a.upper == b.upper).all())

    def _random_ndarray_from_interval(self, interval):
        return np.random.uniform(interval.lower, interval.upper,
                                 interval.shape)

    def _assert_in_interval(self, array, interval):
        self.assertTrue((interval.lower <= array).all())
        self.assertTrue((interval.upper >= array).all())

class TestShape(TestCase):
    def _run_test(self, shape):
        i = NpInterval(np.zeros(shape), np.ones(shape))
        self.assertEquals(shape, i.shape)

    def test_shape(self):
        for i in xrange(100):
            self._run_test(_random_shape())


class TestMultiplying(TestNpInterval):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-42, -5, -1]])
        au = np.asarray([[2, -1,  1], [4, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[1, -4, -5], [-4,  4, -91]])
        ru = np.asarray([[4, -1,  5], [42, 10,  13]])

        R = A * B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

        R = B
        R *= A
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a * b <= ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, B * A)

    def test_shape(self):
        for i in xrange(100):
            shape = self._random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A * B
            self.assertEqual(R.shape, shape)

    def test_random_with_float(self):
        a = self._random_npinterval()
        b = uniform(1., 100.)
        result = a * b
        self.assertTrue((a.lower * b == result.lower).all())
        self.assertTrue((a.upper * b == result.upper).all())
        self._check_lower_upper(result)

        b = uniform(-100., -1.)
        result = a * b
        self.assertTrue((a.lower * b == result.upper).all())
        self.assertTrue((a.upper * b == result.lower).all())
        self._check_lower_upper(result)

    def test_set_with_float(self):
        a = NpInterval(np.array([1., -4., 0., 5., -3.]),
                       np.array([1., -1., 2., 12.5, 3]))
        b = 2.5
        result = a * b
        expected_result = NpInterval(
            np.array([2.5, -10, 0, 12.5, -7.5]),
            np.array([2.5, -2.5, 5, 31.25, 7.5])
        )
        self.assertTrue((result.lower == expected_result.lower).all())
        self.assertTrue((result.upper == expected_result.upper).all())

        b = -2.5
        result = a * b
        expected_result = NpInterval(
            np.array([-2.5, 2.5, -5., -31.25, -7.5]),
            np.array([-2.5, 10, 0, -12.5, 7.5])
        )
        self.assertTrue((result.lower == expected_result.lower).all())
        self.assertTrue((result.upper == expected_result.upper).all())

    def test_random_with_ndarray(self):
        shape = self._random_shape()
        a = self._random_npinterval(shape)
        b = np.full(shape, 6.)
        result = a * b
        self.assertTrue((a.lower * b == result.lower).all())
        self.assertTrue((a.upper * b == result.upper).all())
        self._check_lower_upper(result)

    def test_random_example(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape=shape)
            b = self._random_npinterval(shape=shape)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                b_random = self._random_ndarray_from_interval(b)
                self._assert_in_interval(a_random * b_random, a * b)


class TestAdding(TestNpInterval):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[2, -1, 3], [-5, -7, -14]])
        ru = np.asarray([[4,  1, 6], [41, -5,  8]])

        R = A + B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

        R = B
        R += A
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a + b <= ru)
        self.assertTrue(al + bl - 1 < rl)
        self.assertTrue(au + bu + 1 > ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, B + A)

    def test_shape(self):
        for i in xrange(100):
            shape = self._random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A + B
            self.assertEqual(R.shape, shape)

    def test_random_example(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape=shape)
            b = self._random_npinterval(shape=shape)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                b_random = self._random_ndarray_from_interval(b)
                self._assert_in_interval(a_random + b_random, a + b)


class TestSub(TestNpInterval):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        rl = np.asarray([[-1, -4, -6], [-3, -4, -2]])
        ru = np.asarray([[1, -2, -3], [43, -2, 20]])

        R = A - B
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, B, R):
        al, au = A.lower[0], A.upper[0]
        bl, bu = B.lower[0], B.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a, b in product(xrange(al, au), xrange(bl, bu)):
            self.assertTrue(rl <= a - b <= ru)

    def test_correct(self):
        for i in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            self._check_result(A, B, A - B)

    def test_shape(self):
        for i in xrange(100):
            shape = self._random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A - B
            self.assertEqual(R.shape, shape)

    def test_random_example(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape=shape)
            b = self._random_npinterval(shape=shape)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                b_random = self._random_ndarray_from_interval(b)
                self._assert_in_interval(a_random - b_random, a - b)


class TestSquare(TestNpInterval):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-42, -5, -1]])
        au = np.asarray([[2, -1,  1], [4, -4, 7]])
        A = NpInterval(al, au)

        rl = np.asarray([[1, 1, 0], [0, 16, 0]])
        ru = np.asarray([[4, 4, 1], [1764, 25, 49]])

        R = A.square()
        self.assertTrue((rl == R.lower).all())
        self.assertTrue((ru == R.upper).all())

    def _check_result(self, A, R):
        al, au = A.lower[0], A.upper[0]
        rl, ru = R.lower[0], R.upper[0]
        for a in xrange(al, au):
            self.assertTrue(rl <= a * a <= ru)
        bigger = max(al*al, au*au) + 1
        self.assertTrue(bigger > ru)
        self.assertTrue(rl > -1)

    def test_correct(self):
        for i in xrange(100):
            a, b = randrange(-10, 10), randrange(-10, 10)
            if a > b:
                a, b = b, a
            A = NpInterval(np.asarray([a]), np.asarray([b]))
            self._check_result(A, A.square())

    def test_shape(self):
        for i in xrange(100):
            shape = self._random_shape()
            A = NpInterval(np.ones(shape), 2 * np.ones(shape))
            self.assertEqual(A.square().shape, shape)

    def test_random_example(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape=shape)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                self._assert_in_interval(a_random * a_random, a.square())


class TestGetSetitem(TestNpInterval):
    def test_1D(self):
        n = 100
        I = NpInterval(np.zeros((n,)), np.zeros((n,)))
        for i in xrange(n):
            I[i] = NpInterval(np.asarray([-i]), np.asarray([i]))
        for i in xrange(n):
            self.assertEquals(I[i].lower, -i)
            self.assertEquals(I[i].upper, i)

    def test_2D(self):
        n = 100
        I = NpInterval(np.zeros((n, n)), np.zeros((n, n)))
        for i, j in product(xrange(n), xrange(n)):
            I[i][j] = NpInterval(np.asarray([i ^ (j**2) - 42]),
                                 np.asarray([i**2 + j**3 / 7]))
        for i, j in product(xrange(n), xrange(n)):
            self.assertEquals(I[i][j].lower, i ^ (j**2) - 42)
            self.assertEquals(I[i][j].upper, i**2 + j**3 / 7)

    def test_3D(self):
        n = 10
        I = NpInterval(np.zeros((n, n, n)), np.zeros((n, n, n)))
        for i, j, k in product(xrange(n), xrange(n), xrange(n)):
            I[i][j][k] = NpInterval(np.asarray([i + j - k ^ 67]),
                                    np.asarray([i * j + 42 * k]))
        for i, j, k in product(xrange(n), xrange(n), xrange(n)):
            self.assertEquals(I[i][j][k].lower, i + j - k ^ 67)
            self.assertEquals(I[i][j][k].upper, i * j + 42 * k)

    def test_4D(self):
        n = 10
        I = NpInterval(np.zeros((n, n, n, n)), np.zeros((n, n, n, n)))
        for i, j, k, l in product(xrange(n), xrange(n), xrange(n), xrange(n)):
            I[i][j][k][l] = NpInterval(np.asarray([i*l ^ j*k]),
                                       np.asarray([(i*j ^ l*k) + 1000]))
        for i, j, k, l in product(xrange(n), xrange(n), xrange(n), xrange(n)):
            self.assertEquals(I[i][j][k][l].lower, i*l ^ j*k)
            self.assertEquals(I[i][j][k][l].upper, (i*j ^ l*k) + 1000)


class TestAntiadd(TestCase):
    def test_case(self):
        al = np.asarray([[1, -2, -1], [-4, -5, -1]])
        au = np.asarray([[2, -1,  1], [42, -4, 7]])
        A = NpInterval(al, au)

        bl = np.asarray([[1, 1, 4], [-1, -2, -13]])
        bu = np.asarray([[2, 2, 5], [-1, -1,   1]])
        B = NpInterval(bl, bu)

        R = (A + B).antiadd(B)
        self.assertTrue((A.lower == R.lower).all())
        self.assertTrue((A.upper == R.upper).all())
        R = (A + B).antiadd(A)
        self.assertTrue((B.lower == R.lower).all())
        self.assertTrue((B.upper == R.upper).all())

    def test_correct(self):
        for _ in xrange(100):
            l = [randrange(-10, 10) for j in xrange(4)]
            A = NpInterval(np.asarray([l[0]]), np.asarray([l[1]]))
            B = NpInterval(np.asarray([l[2]]), np.asarray([l[3]]))

            if A.lower[0] > A.upper[0]:
                A.lower, A.upper = A.upper, A.lower
            if B.lower[0] > B.upper[0]:
                B.lower, B.upper = B.upper, B.lower

            R = (A + B).antiadd(B)
            self.assertTrue((A.lower == R.lower).all())
            self.assertTrue((A.upper == R.upper).all())
            R = (A + B).antiadd(A)
            self.assertTrue((B.lower == R.lower).all())
            self.assertTrue((B.upper == R.upper).all())

    def test_shape(self):
        for i in xrange(100):
            shape = _random_shape()
            A = NpInterval(np.ones(shape), 100 * np.ones(shape))
            B = NpInterval(np.ones(shape) * 2, np.ones(shape) * 3)
            R = A.antiadd(B)
            self.assertEqual(R.shape, shape)


class TestDNorm(TestCase):

    def foo(self, x, c, a, b):
        return (a * (1 - 2 * b) * x ** 2 + c) / (a * x ** 2 + c) ** (b + 1)

    def foo2(self, x, y, c, a, b):
        return -2 * a * b * x * y * ((a * (x ** 2 + y ** 2) + c) ** (-b-1))

    def test_case0(self):
        a = 1.
        b = 0.75
        k = 1.
        # local range = 1
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        res = self._count_norm(act, der, k, a, b, 1)
        R = derivative.op_d_norm(activation, act.shape, 1, k, a, b)
        self.assertTrue(np.isclose(abs(res), R.upper).all())
        self.assertTrue(np.isclose(-abs(res), R.lower).all())

        # local range = 5
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0]**2
        res = np.zeros(act.shape)
        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b)
            else:
                res[:, j, ::] += self.foo2(x, y, c - x**2 - y**2, a, b)

        res2 = self._count_norm(act, der, k, a, b, 5)
        R = derivative.op_d_norm(activation, act.shape, 5, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def test_case1(self):
        a = 1.
        b = 0.75
        k = 1.
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        def foo(x, c):
            return (a * (1 - 2 * b) * x ** 2 + c) / (a * x ** 2 + c) ** (b + 1)

        def foo2(x, y, c):
            return -2 * a * b * x * y * (
                (a * (x ** 2 + y ** 2) + c) ** (-b - 1))

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)

        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += foo(x, c - x ** 2)
            else:
                res[:, i, ::] += foo2(x, y, c - x ** 2 - y ** 2)

        R = derivative.op_d_norm(activation, act.shape, 5, k, a, b)
        self.assertTrue((R.lower <= -abs(res)).all())
        self.assertTrue((abs(res) <= R.upper).all())

    def test_case2(self):
        a = 4.
        b = 3
        k = 0.8
        # local range = 1
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[1.]], [[1.]], [[1.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(-der, der)

        res = self.foo(act, k, a, b)
        R = derivative.op_d_norm(activation, act.shape, 1, k, a, b)

        self.assertTrue(np.isclose(-res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

        # local range = 5
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)

        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b)
            else:
                res[:, j, ::] += self.foo2(x, y, c - x ** 2 - y ** 2, a, b)

        res2 = self._count_norm(act, der, k, a, b, 5)
        R = derivative.op_d_norm(activation, act.shape, 5, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def test_case3(self):
        a = 1.
        b = 0.75
        k = 1.
        # local range = 1
        act = np.asarray([[[[2.]], [[3.]], [[5.]]]])
        der = np.asarray([[[[-3.]], [[2.]], [[7.]]]])
        activation = NpInterval(act, 1 * act)
        derivative = NpInterval(der, 1 * der)

        res = self._count_norm(act, der, k, a, b, 1)
        R = derivative.op_d_norm(activation, act.shape, 1, k, a, b)
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

        # local range = 5
        derivative = NpInterval(1 * der, 1 * der)

        c = k
        for i in xrange(3):
            c += act[0][i][0][0] ** 2
        res = np.zeros(act.shape)
        for i, j in product(xrange(3), xrange(3)):
            x = act[0][i][0][0]
            y = act[0][j][0][0]
            if i == j:
                res[:, i, ::] += self.foo(x, c - x ** 2, a, b) * der[:, i, ::]
            else:
                res[:, j, ::] += \
                    self.foo2(x, y, c - x ** 2 - y ** 2, a, b) * der[:, i, ::]

        res2 = self._count_norm(act, der, k, a, b, 5)
        R = derivative.op_d_norm(activation, act.shape, 5, k, a, b)
        self.assertTrue(np.isclose(res, res2).all())
        self.assertTrue(np.isclose(res, R.upper).all())
        self.assertTrue(np.isclose(res, R.lower).all())

    def _count_norm(self, act, der, k, alpha, beta, local_range):
        res = np.zeros_like(act)
        b, ch, h, w = der.shape
        local_range /= 2
        for at_b, at_ch, at_h, at_w in product(xrange(b), xrange(ch),
                                               xrange(h), xrange(w)):
            c = k
            y = act[at_b, at_ch, at_h, at_w]
            for i in xrange(-local_range, local_range + 1):
                if i != 0 and 0 <= (at_ch + i) < ch:
                    c += act[at_b, at_ch + i, at_h, at_w]**2

            res[at_b, at_ch, at_h, at_w] += \
                self.foo(y, c, alpha, beta) * der[at_b, at_ch, at_h, at_w]

            for i in xrange(-local_range, local_range + 1):
                if i != 0 and 0 <= at_ch + i < ch:
                    x = act[at_b, at_ch + i, at_h, at_w]
                    c -= x**2
                    res[at_b, at_ch + i, at_h, at_w] += \
                        self.foo2(x, y, c, alpha, beta) * der[at_b, at_ch,
                                                              at_h, at_w]
                    c += x**2
        return res

    def test_correct(self):
        for _ in xrange(100):
            s = randrange(1, 10)
            shape = (1, s, 1, 1)
            local_range = randrange(1, 3) * 2 + 1
            k = uniform(1, 10)
            a = uniform(1, 10)
            b = uniform(0.75, 3)
            activations = _random_npinterval(shape)
            derivatives = _random_npinterval(shape)
            R = derivatives.op_d_norm(activations, shape, local_range,
                                      k, a, b)
            for _ in xrange(100):
                act = _rand_from_npinterval(activations)
                der = _rand_from_npinterval(derivatives)
                res = self._count_norm(act, der, k, a, b, local_range)
                self.assertTrue((R.lower <= res).all())
                self.assertTrue((res <= R.upper).all())

    def test_correct_flat(self):
        for _ in xrange(100):
            s = randrange(1, 10)
            shape = (1, s, 1, 1)
            local_range = randrange(1, 3) * 2 + 1
            k = uniform(1, 10)
            a = uniform(1, 10)
            b = uniform(0.75, 3)
            activations = _random_npinterval(shape)
            activations.lower = activations.upper * 1
            derivatives = _random_npinterval(shape)
            derivatives.lower = derivatives.upper * 1
            R = derivatives.op_d_norm(activations, shape, local_range,
                                      k, a, b)
            act = _rand_from_npinterval(activations)
            der = _rand_from_npinterval(derivatives)
            res = self._count_norm(act, der, k, a, b, local_range)

            self.assertTrue(np.isclose(R.lower, res).all())
            self.assertTrue(np.isclose(res, R.upper).all())

    def test_shape(self):
        for _ in xrange(20):
            shape = _random_shape(4, limit=100)
            local_range = randrange(0, 3)
            k = uniform(0.1, 10)
            a = uniform(0.1, 10)
            b = uniform(0.75, 3)

            A = _random_npinterval(shape)
            D = _random_npinterval(shape)
            R = D.op_d_norm(A, shape, local_range, k, a, b)
            self.assertEquals(A.shape, R.shape)


class ReluDerivativeTest(TestCase):

    def test_case1(self):
        shp = (4, 3, 2)
        act = np.zeros(shp)
        for n_in in range(4):
            for h in range(3):
                for w in range(2):
                    act[n_in, h, w] += 100 * n_in + 10 * h + w
        iact = NpInterval(act, 1 * act)
        idout = NpInterval(np.ones(shp), np.ones(shp))
        idin = idout.op_d_relu(iact)
        l, u = idin.lower, idin.upper

        self.assertAlmostEquals(l[0, 0, 0], 0.0)
        self.assertAlmostEquals(u[0, 0, 0], 0.0)
        self.assertAlmostEquals(l[2, 1, 1], 1.0)
        self.assertAlmostEquals(l[2, 2, 1], 1.0)
        self.assertAlmostEquals(l[1, 0, 1], 1.0)
        self.assertAlmostEquals(l[2, 1, 1], 1.0)
        self.assertAlmostEquals(l[2, 2, 0], 1.0)
        self.assertAlmostEquals(l[1, 0, 1], 1.0)

    def test_case2(self):
        actl = np.asarray([-2, -1, -1, 0, 0, 1])
        actu = np.asarray([-1, 1, 0, 0, 1, 2])
        doutl = np.asarray([2, 3, 4, 7, 11, 13])
        doutu = np.asarray([3, 5, 7, 11, 13, 17])
        iact = NpInterval(actl, actu)
        idout = NpInterval(doutl, doutu)
        idin = idout.op_d_relu(iact)
        l, u = idin.lower, idin.upper
        rl = np.asarray([0, 0, 0, 0, 11, 13])
        ru = np.asarray([0, 5, 0, 0, 13, 17])

        self.assertTrue((l == rl).all())
        self.assertTrue((u == ru).all())

    def test_case3(self):
        actl = np.asarray([-2, -1, -1, 0, 0, 1])
        actu = np.asarray([-1, 1, 0, 0, 1, 2])
        doutl = np.asarray([-3, -5, -7, -11, -13, -17])
        doutu = np.asarray([-2, -3, -5, -7, -11, -13])
        iact = NpInterval(actl, actu)
        idout = NpInterval(doutl, doutu)
        idin = idout.op_d_relu(iact)
        l, u = idin.lower, idin.upper
        rl = np.asarray([0, -5, 0, 0, -13, -17])
        ru = np.asarray([0, 0, 0, 0, -11, -13])

        self.assertTrue((l == rl).all())
        self.assertTrue((u == ru).all())


class ConvDerivativeTest(TestCase):
    def test_case1(self):
        input_shape = (1, 1, 3, 5)
        derivatives = np.asarray([1, -1]).reshape((1, 1, 1, 2))
        D = NpInterval(derivatives, 1 * derivatives)
        filter_shape = (1, 3, 3)
        filter = np.asarray([[1, 2, 4], [-1, -2, -4], [0, 0, 0]]).reshape(
            (1, 1, 3, 3))
        A = D.op_d_conv(input_shape, filter_shape, filter, (1, 2), (0, 0), 1)

        result = np.asarray([[[[0., 0., 0., 0., 0.],
                               [-4., -2., 3., 2., 1.],
                               [4., 2., -3., -2., -1.]]]])
        self.assertEquals(A.shape, result.shape)
        self.assertTrue((A.upper == result).all())
        self.assertTrue((A.lower == result).all())

    def test_case2(self):
        input_shape = (1, 1, 3, 5)
        derivatives = np.asarray([[1, 0], [0, -1], [1, -1]]).\
            reshape((1, 1, 3, 2))
        D = NpInterval(derivatives, 1 * derivatives)
        filter_shape = (1, 1, 3)
        filter = np.asarray([1, 2, 4]).reshape((1, 1, 1, 3))
        A = D.op_d_conv(input_shape, filter_shape, filter, (1, 2), (0, 0), 1)

        result = np.asarray([[[[4., 2., 1., 0., 0.],
                               [0., 0., -4., -2., -1.],
                               [4., 2., -3., -2., -1.]]]])
        self.assertEquals(A.shape, result.shape)
        self.assertTrue((A.upper == result).all())
        self.assertTrue((A.lower == result).all())

    def test_dims(self):
        derivarives = np.ones((1, 2, 2, 4))
        D = NpInterval(derivarives, 1 * derivarives)
        w = np.ones((2, 1, 3, 4))
        w = w[:, :, ::-1, ::-1]
        R = D.op_d_conv((1, 1, 4, 7), (2, 3, 4), w, stride=(1, 1),
                        padding=(0, 0), n_groups=1)
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, u)
        assert_array_almost_equal(l, np.asarray(
            [[[[2, 4, 6, 8, 6, 4, 2], [4, 8, 12, 16, 12, 8, 4],
               [4, 8, 12, 16, 12, 8, 4], [2, 4, 6, 8, 6, 4, 2]]]]))

    def test_2x2_float(self):
        derivatives = np.asarray([[[[4, 8], [2, 3]]]])
        D = NpInterval(derivatives, 1 * derivatives)
        w = np.asarray([[[[2, 3, 0], [5, 7, 0], [0, 0, 0]]]])
        w = w[:, :, ::-1, ::-1]
        R = D.op_d_conv((1, 1, 2, 2), (1, 3, 3), w, padding=(1, 1),
                        stride=(1, 1), n_groups=1)
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, u)
        assert_array_almost_equal(l, np.asarray([[[[80, 65], [29, 21]]]]))

    def test_all_dims(self):
        derivatives = np.asarray([[[[2, 3], [5, 7]],
                                   [[0.2, 0.3], [0.5, 0.7]]]])
        D = NpInterval(derivatives, 1 * derivatives)
        w = np.asarray([[[[1, 0, 2], [0, 4, 0], [3, 0, 0]],
                         [[0, 0, 0], [0, 9, 10], [0, 11, 12]]],
                        [[[5, 0, 6], [0, 0, 0], [7, 0, 8]],
                         [[13, 15, 0], [0, 0, 0], [14, 16, 0]]]])
        w = w[:, :, ::-1, ::-1]
        R = D.op_d_conv((1, 2, 2, 2), (2, 3, 3), w, padding=(1, 1),
                        stride=(1, 1), n_groups=1)
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, u)
        assert_array_almost_equal(l, np.asarray([[[[18.5, 25], [31.1, 29.6]],
                                                  [[34.6, 57.5],
                                                   [74.4, 174.8]]]]))

    def test_interval_behavior(self):
        derivatives = np.asarray([1, 1]).reshape(1, 1, 1, 2)
        w = np.asarray([[1, -10, 3], [-2, 5, 6]]).reshape((1, 1, 2, 3))
        stride = (1, 2)
        image_size = (1, 1, 2, 5)
        filter_shape = (1, 2, 3)
        D = NpInterval(1 * derivatives, derivatives)
        R = D.op_d_conv(image_size, filter_shape, w, stride, (0, 0), 1)
        res = np.asarray([[[[6., 5., 4., 5., -2.],
                            [3., -10., 4., -10., 1.]]]])
        assert_array_almost_equal(res, R.lower)
        assert_array_almost_equal(res, R.upper)

        D = NpInterval(-derivatives, derivatives)
        R = D.op_d_conv(image_size, filter_shape, w, stride, (0, 0), 1)
        res = np.asarray([[[[6., 5., 8., 5., 2.],
                            [3., 10., 4., 10., 1.]]]])
        assert_array_almost_equal(-res, R.lower)
        assert_array_almost_equal(res, R.upper)

    def test_correct(self):
        input_shape = (2, 3, 3, 5)
        w_shape = (1, 3, 2, 2)
        filter_shape = (1, 2, 2)
        der_shape = (2, 1, 4, 6)
        stride = (1, 1)
        padding = (1, 1)

        for _ in xrange(10):
            D = _random_npinterval(der_shape)
            w = np.random.rand(*w_shape)
            R = D.op_d_conv(input_shape, filter_shape, w, stride, padding, 1)
            for i in xrange(100):
                der_val = _rand_from_npinterval(D)
                d = NpInterval(der_val, 1 * der_val)
                r = d.op_d_conv(input_shape, filter_shape, w, stride,
                                padding, 1)
                self.assertTrue((R.lower <= r.lower).all())
                self.assertTrue((R.upper >= r.upper).all())


class AvgPoolDerivativeTest(TestCase):

    def test_simple(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        I = NpInterval(inp_l, inp_u)
        der_l = np.asarray([[[[-1, -2], [-3, -4]]]])
        der_u = np.asarray([[[[5, 4], [3, 2]]]])
        D = NpInterval(der_l, der_u)
        shp = (1, 1, 3, 3)
        R = D.op_d_avg_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, -3, -2],
                                                   [-4, -10, -6],
                                                   [-3, -7, -4]]]]) / 4.0)
        assert_array_almost_equal(u, np.asarray([[[[5, 9, 4],
                                                   [8, 14, 6],
                                                   [3, 5, 2]]]]) / 4.0)

    def test_channels_batch(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [3, 0, 0]]],
                           [[[0, 3, 3], [4, 5, 6], [7, 8, 4]],
                            [[-3, -3, -3], [-3, -3, -3], [3, 3, 3]]]])

        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                            [[1, 1, 1], [1, 1, 1], [4, 1, 1]]],
                           [[[2, 4, 4], [9, 9, 9], [9, 9, 9]],
                            [[2, 2, 2], [2, 2, 2], [5, 5, 5]]]])

        I = NpInterval(inp_l, inp_u)

        der_l = np.asarray([[[[-1, -2], [-3, -4]],
                             [[1, 2], [-3, -2]]],
                            [[[1, 2], [-3, -2]],
                             [[-1, 1], [-1, 1]]]])

        der_u = np.asarray([[[[5, 4], [3, 2]],
                             [[4, 4], [4, 4]]],
                            [[[4, 5], [0, 1]],
                             [[0, 2], [0, 2]]]])

        D = NpInterval(der_l, der_u)
        shp = (2, 2, 3, 3)

        R = D.op_d_avg_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper

        assert_array_almost_equal(l, np.asarray([[[[-1, -3, -2], [-4, -10, -6],
                                                   [-3, -7, -4]],
                                                  [[1, 3, 2], [-2, -2, 0],
                                                   [-3, -5, -2]]],
                                                 [[[1, 3, 2], [-2, -2, 0],
                                                   [-3, -5, -2]],
                                                  [[-1, 0, 1], [-2, 0, 2],
                                                   [-1, 0, 1]]]]) / 4.0)
        assert_array_almost_equal(u, np.asarray([[[[5, 9, 4], [8, 14, 6],
                                                   [3, 5, 2]],
                                                  [[4, 8, 4], [8, 16, 8],
                                                   [4, 8, 4]]],
                                                 [[[4, 9, 5], [4, 10, 6],
                                                   [0, 1, 1]],
                                                  [[0, 2, 2], [0, 4, 4],
                                                   [0, 2, 2]]]]) / 4.0)

    def test_stride(self):
        inp_l = np.arange(25).reshape((1, 1, 5, 5))
        inp_u = np.arange(25).reshape((1, 1, 5, 5)) + 2
        I = NpInterval(inp_l, inp_u)
        derivatives = np.asarray([[[[-1, 2], [-3, 4]]]])
        D = NpInterval(derivatives, 1 * derivatives)
        shp = (1, 1, 5, 5)
        R = D.op_d_avg_pool(I, shp, poolsize=(2, 2), stride=(3, 3),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, -1, 0, 2, 2],
                                                   [-1, -1, 0, 2, 2],
                                                   [0, 0, 0, 0, 0],
                                                   [-3, -3, 0, 4, 4],
                                                   [-3, -3, 0, 4, 4]]]]) / 4.0)
        assert_array_almost_equal(u, np.asarray([[[[-1, -1, 0, 2, 2],
                                                   [-1, -1, 0, 2, 2],
                                                   [0, 0, 0, 0, 0],
                                                   [-3, -3, 0, 4, 4],
                                                   [-3, -3, 0, 4, 4]]]]) / 4.0)

    def test_padding(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        I = NpInterval(inp_l, inp_u)
        der_l = np.asarray([[[[-1, -2], [-3, -4]]]])
        der_u = np.asarray([[[[5, 4], [3, 2]]]])
        D = NpInterval(der_l, der_u)
        shp = (1, 1, 3, 3)

        R = D.op_d_avg_pool(I, shp, poolsize=(2, 2), padding=(1, 1),
                            stride=(3, 3))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, 0, -2], [0, 0, 0],
                                                   [-3, 0, -4]]]]) / 4.0)
        assert_array_almost_equal(u, np.asarray([[[[5, 0, 4], [0, 0, 0],
                                                   [3, 0, 2]]]]) / 4.0)


class MaxPoolDerivativeTest(TestCase):

    def test_simple(self):
        inp_l = np.asarray([[[[1, 1], [1, 1]]]])
        inp_u = np.asarray([[[[2, 2], [2, 2]]]])
        I = NpInterval(inp_l, inp_u)
        derivatives = np.asarray([[[[5]]]])
        D = NpInterval(derivatives, 1 * derivatives)
        shp = (1, 1, 2, 2)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[0, 0], [0, 0]]]]))
        assert_array_almost_equal(u, np.asarray([[[[5, 5], [5, 5]]]]))

    def test_neg_output(self):
        inp_l = np.asarray([[[[1, 1], [1, 1]]]])
        inp_u = np.asarray([[[[2, 2], [2, 2]]]])
        I = NpInterval(inp_l, inp_u)
        derivative = np.asarray([[[[-3]]]])
        D = NpInterval(derivative, 1 * derivative)
        shp = (1, 1, 2, 2)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-3, -3], [-3, -3]]]]))
        assert_array_almost_equal(u, np.asarray([[[[0, 0], [0, 0]]]]))

    def test_2D(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        I = NpInterval(inp_l, inp_u)
        der_l = np.asarray([[[[-1, -2], [-3, -4]]]])
        der_u = np.asarray([[[[5, 4], [3, 2]]]])
        D = NpInterval(der_l, der_u)
        shp = (1, 1, 3, 3)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, -3, -2], [-4, -10, -6],
                                                   [-3, -7, -4]]]]))
        assert_array_almost_equal(u, np.asarray([[[[5, 9, 4], [8, 14, 6],
                                                   [3, 5, 2]]]]))

    def test_channels_batch(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[0, 0, 0], [0, 0, 0], [3, 0, 0]]],
                            [[[0, 3, 3], [4, 5, 6], [7, 8, 4]],
                             [[-3, -3, -3], [-3, -3, -3], [3, 3, 3]]]])
        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             [[1, 1, 1], [1, 1, 1], [4, 1, 1]]],
                            [[[2, 4, 4], [9, 9, 9], [9, 9, 9]],
                             [[2, 2, 2], [2, 2, 2], [5, 5, 5]]]])
        I = NpInterval(inp_l, inp_u)

        der_l = np.asarray([[[[-1, -2], [-3, -4]],
                             [[1, 2], [-3, -2]]],
                            [[[1, 2], [-3, -2]],
                             [[-1, 1], [-1, 1]]]])
        der_u = np.asarray([[[[5, 4], [3, 2]],
                             [[4, 4], [4, 4]]],
                            [[[4, 5], [0, 1]],
                             [[0, 2], [0, 2]]]])
        D = NpInterval(der_l, der_u)
        shp = (2, 2, 3, 3)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), stride=(1, 1),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, -3, -2], [-4, -10, -6],
                                                   [-3, -7, -4]],
                                                  [[0, 0, 0], [0, -2, -2],
                                                   [-3, -2, -2]]],
                                                 [[[0, 0, 0], [-3, -5, -2],
                                                   [-3, -5, -2]],
                                                  [[-1, -1, 0], [-1, -1, 0],
                                                   [-1, -1, 0]]]]))
        assert_array_almost_equal(u, np.asarray([[[[5, 9, 4], [8, 14, 6],
                                                   [3, 5, 2]],
                                                  [[4, 8, 4], [4, 12, 8],
                                                   [4, 4, 4]]],
                                                 [[[0, 0, 0], [4, 10, 6],
                                                   [0, 1, 1]],
                                                  [[0, 2, 2], [0, 2, 2],
                                                   [0, 2, 2]]]]))

    def test_stride(self):
        tinp_l = np.arange(25).reshape((1, 1, 5, 5))
        tinp_u = np.arange(25).reshape((1, 1, 5, 5)) + 2
        I = NpInterval(tinp_l, tinp_u)
        derivative = np.asarray([[[[-1, 2], [-3, 4]]]])
        D = NpInterval(derivative, 1 * derivative)
        shp = (1, 1, 5, 5)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), stride=(3, 3),
                            padding=(0, 0))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[0, 0, 0, 0, 0],
                                                   [-1, -1, 0, 0, 0],
                                                   [0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0],
                                                   [-3, -3, 0, 0, 0]]]]))
        assert_array_almost_equal(u, np.asarray([[[[0, 0, 0, 0, 0],
                                                   [0, 0, 0, 2, 2],
                                                   [0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0],
                                                   [0, 0, 0, 4, 4]]]]))

    def test_padRg(self):
        inp_l = np.asarray([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
        inp_u = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        I = NpInterval(inp_l, inp_u)
        der_l = np.asarray([[[[-1, -2], [-3, -4]]]])
        der_u = np.asarray([[[[5, 4], [3, 2]]]])
        D = NpInterval(der_l, der_u)
        shp = (1, 1, 3, 3)

        R = D.op_d_max_pool(I, shp, poolsize=(2, 2), padding=(1, 1),
                            stride=(3, 3))
        l, u = R.lower, R.upper
        assert_array_almost_equal(l, np.asarray([[[[-1, 0, -2], [0, 0, 0],
                                                   [-3, 0, -4]]]]))
        assert_array_almost_equal(u, np.asarray([[[[5, 0, 4], [0, 0, 0],
                                                   [3, 0, 2]]]]))


class TestDiv(TestNpInterval):

    def _random_npinterval_without_zeros(self, shape=None, size_limit=10**2,
                                         number_limit=10**2):
        if shape is None:
            shape = self._random_shape(size_limit, 4)
        sign = np.select([np.random.rand(*shape) > 0.5, True], [1, -1])
        a = np.random.rand(*shape) * uniform(1, number_limit) * sign
        b = np.random.rand(*shape) * uniform(1, number_limit) * sign
        return NpInterval(np.minimum(a, b), np.maximum(a, b))

    def test_div_random_with_float(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            b = uniform(-100., 100.)

            if b == 0:
                continue

            result = a / b
            if b > 0:
                self.assertTrue((a.lower / b == result.lower).all())
                self.assertTrue((a.upper / b == result.upper).all())
            else:
                self.assertTrue((a.lower / b == result.upper).all())
                self.assertTrue((a.upper / b == result.lower).all())
            self._check_lower_upper(result)

    def test_rdiv_random_with_float(self):
        for _ in xrange(20):
            a = uniform(-100., 100.)
            b = self._random_npinterval_without_zeros()

            result = a / b
            if a < 0:
                self.assertTrue((a / b.lower == result.lower).all())
                self.assertTrue((a / b.upper == result.upper).all())
            else:
                self.assertTrue((a / b.lower == result.upper).all())
                self.assertTrue((a / b.upper == result.lower).all())
            self._check_lower_upper(result)

    def test_div_random_with_ndarray(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape)
            b = self._random_ndarray(shape)

            if (b == 0).any():
                continue

            result = a / b
            self._check_lower_upper(result)

    def test_div_with_float(self):
        a = NpInterval(
            np.array([[2., 1.], [-5., -10.]]),
            np.array([[4., 10.], [1., -8]])
        )
        b = -2.

        result = a / b
        expected_result = NpInterval(
            np.array([[-2., -5.], [-0.5, 4.]]),
            np.array([[-1, -0.5], [2.5, 5.]])
        )
        self._assert_npintervals_equal(result, expected_result)

    def test_rdiv_with_float(self):
        a = 2.
        b = NpInterval(
            np.array([[2., 1.], [-5., -10.]]),
            np.array([[4., 10.], [-1., -8]])
        )

        result = a / b
        expected_result = NpInterval(
            np.array([[0.5, 0.2], [-2., -0.25]]),
            np.array([[1, 2.], [-0.4, -0.2]])
        )
        self._assert_npintervals_equal(result, expected_result)

    def test_div_random(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape)
            b = self._random_npinterval_without_zeros(shape)

            result = a / b
            self._check_lower_upper(result)

    def test_div(self):
        a = NpInterval(
            np.array([[1., -3.], [5., -10.]]),
            np.array([[1., -1.], [12., 10.]])
        )
        b = NpInterval(
            np.array([[0.1, 2.], [-2., 0.5]]),
            np.array([[2., 2.], [-1., 1.]])
        )

        result = a / b
        expected_result = NpInterval(
            np.array([[0.5, -1.5], [-12., -20]]),
            np.array([[10., -0.5], [-2.5, 20]])
        )
        self._assert_npintervals_equal(result, expected_result)

    def test_div_random_example(self):
        for _ in xrange(20):
            shape = self._random_shape()
            a = self._random_npinterval(shape=shape)
            b = self._random_npinterval_without_zeros(shape=shape)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                b_random = self._random_ndarray_from_interval(b)
                self._assert_in_interval(a_random / b_random, a / b)


class TestPower(TestNpInterval):
    def pow_to_zero(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            result = a.power(0.)
            self.assertTrue((a == 1.).all())

    def pow_to_one(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            self._assert_npintervals_equal(a, a.power(1.))

    def pow_to_minus_ones(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            self._assert_npintervals_equal(a.power(-1.), 1. / a)

    def pow_to_positive_integer(self):
        a = NpInterval(np.array([-1, 0, 3, -4]), np.array([2, 1, 4, -2]))
        expected_odd_result = NpInterval(np.array([-1, 0, 27, -64]),
                                         np.array([8, 1, 64, -8]))
        self._assert_npintervals_equal(a.pow(3), expected_odd_result)

        expected_even_result = NpInterval(np.array([0, 0, 9, 4]),
                                          np.array([4, 1, 16, 16]))
        self._assert_npintervals_equal(a.pow(2), expected_even_result)

    def pow_positive_to_fraction(self):
        for _ in xrange(20):
            a_1 = np.random.uniform(0.001, 10**3, 100)
            a_2 = np.random.uniform(0.001, 10**3, 100)
            a = NpInterval(np.minimum(a_1, a_2), np.maximum(a_1, a_2))
            b = uniform(0.001, 10**3)
            result = a.pow(b)
            self.assertTrue((result.lower == a.lower.pow(b)).all())
            self.assertTrue((result.upper == a.upper.pow(b)).all())

    def pow_to_negative_integer(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            b = randint(1, 10**3)
            self._assert_npintervals_equal(a.pow(-b), (1. / a).pow(b))

    @expectedFailure
    def pow_negative_to_fraction(self):
        a = NpInterval(np.array([-5]), np.array([-1]))
        a.pow(0.5)

    def test_random_example(self):
        for _ in xrange(20):
            a = self._random_npinterval(shape=(1, ))
            b = randint(2, 10**3)
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                self._assert_in_interval(np.power(a_random, b), a.power(b))


class TestDot(TestNpInterval):
    def test_dot_simple(self):
        a = NpInterval(np.array([[1, 2]]), np.array([[3, 4]]))
        b = np.array([[1], [1]])
        expexted_result = NpInterval(np.array([3]), np.array([7]))
        self._assert_npintervals_equal(a.dot(b), expexted_result)

    def test_dot(self):
        a = NpInterval(np.array([[1, -3, 5, -7], [0, 3, -1, -3]]),
                       np.array([[3, -1, 9, 0], [3.5, 3, -0.5, 2]]))
        b = np.array([[2, -1], [1, 3], [-1, 0], [0, 2]])
        expected_result = NpInterval(np.array([[-10, -26], [3.5, -0.5]]),
                                     np.array([[0, -4], [11, 13]]))
        self._assert_npintervals_equal(a.dot(b), expected_result)

    def test_dot_random(self):
        for _ in xrange(20):
            a_np = self._random_ndarray(shape=(5, 8))
            a = NpInterval(a_np, a_np)
            b = self._random_ndarray(shape=(8, 5))
            expected_result = NpInterval(a_np.dot(b), a_np.dot(b))
            self._assert_npintervals_equal(a.dot(b), expected_result)

    def test_dot_check_random_example(self):
        for _ in xrange(20):
            a = self._random_npinterval(shape=(8, 12))
            b = self._random_ndarray(shape=(12, 8))
            for _ in xrange(20):
                a_random = self._random_ndarray_from_interval(a)
                self._assert_in_interval(a_random.dot(b), a.dot(b))


class TestMax(TestNpInterval):
    def test_max(self):
        a = NpInterval(np.array([1, -4, 6, -1, -14]),
                       np.array([13, -1, 6, 0, 198]))
        b = NpInterval(np.array([0, -3, -1, 1, -34]),
                       np.array([0, -1, 5, 19, -1]))
        expected_max = NpInterval(np.array([1, -3, 6, 1, -14]),
                                  np.array([13, -1, 6, 19, 198]))
        self._assert_npintervals_equal(a.max(b), expected_max)
        self._assert_npintervals_equal(b.max(a), expected_max)

    def test_amax(self):
        a = NpInterval(np.array([1, -4, 6, -1, -14]),
                       np.array([13, -1, 6, 0, 198]))
        expected_result = NpInterval(np.array([6]), np.array([198]))
        self._assert_npintervals_equal(a.amax(), expected_result)

    def test_amax_random(self):
        a = self._random_npinterval()
        result = a.amax()
        self.assertTrue((a.lower <= result.lower).all())
        self.assertTrue((a.upper <= result.upper).all())


class TestSmallFunctions(TestNpInterval):
    def test_flatten_random(self):
        for _ in xrange(20):
            shape = self._random_shape()
            size = np.prod(shape)
            a = self._random_npinterval(shape)

            self.assertTrue(a.flatten().lower.size == size)
            self._check_lower_upper(a.flatten())

    def test_flatten(self):
        a = NpInterval(np.array([[2, 3], [4, 5]]), np.array([[6, 7], [9, 9]]))
        result = a.flatten()
        expected_result = NpInterval(np.array([2, 3, 4, 5]),
                                     np.array([6, 7, 9, 9]))
        self._assert_npintervals_equal(result, expected_result)

    def test_sum_random(self):
        a = self._random_npinterval()
        lower_sum = a.lower.sum()
        upper_sum = a.upper.sum()
        expected_result = NpInterval(np.array([lower_sum]),
                                     np.array([upper_sum]))
        self._assert_npintervals_equal(a.sum(), expected_result)
        self._check_lower_upper(a.sum())

    def test_sum(self):
        a = NpInterval(np.array([-1, 3, -10]), np.array([4, 5, 123]))
        expected_result = NpInterval(np.array([-8]), np.array([132]))
        self._assert_npintervals_equal(a.sum(), expected_result)

    def test_neg_random(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            self._assert_npintervals_equal(a, a.neg().neg())

    def test_neg(self):
        a = NpInterval(np.array([5, -3, 0, -12]), np.array([6, -1, 0, 18]))
        expected_result = NpInterval(np.array([-6, 1, 0, -18]),
                                     np.array([-5, 3, 0, 12]))
        self._assert_npintervals_equal(a.neg(), expected_result)

    def test_reshape(self):
        a = self._random_npinterval(shape=(4, 5, 6))
        result = a.reshape((2, 6, 5, 2, 1))
        self.assertTrue(result.shape == (2, 6, 5, 2, 1))

    def test_abs_random(self):
        for _ in xrange(20):
            a = self._random_npinterval()
            self.assertTrue((a.abs().lower >= 0).all())
            self._check_lower_upper(a)

    def test_abs(self):
        a = NpInterval(np.array([-3, 0, -2]), np.array([-1, 3, 1]))
        expexted_result = NpInterval(np.array([1, 0, 0]), np.array([3, 3, 2]))
        self._assert_npintervals_equal(a.abs(), expexted_result)

    def test_T_random(self):
        a = self._random_npinterval(shape=(8, 5))
        self.assertTrue(a.T.shape == (5, 8))
        self._assert_npintervals_equal(a[2, 1], a.T[1, 2])

    def test_T(self):
        a = NpInterval(np.array([[1, 2], [3, 4]]), np.array([[6, 7], [8, 9]]))
        expected_result = NpInterval(np.array([[1, 3], [2, 4]]),
                                     np.array([[6, 8], [7, 9]]))
        self._assert_npintervals_equal(a.T, expected_result)

    def test_derest_output(self):
        for size in xrange(2, 40):
            output = NpInterval.derest_output(size)
            self.assertTrue(output.shape == (size, size))

            self.assertTrue((output.lower.sum(0) == 1).all())
            self.assertTrue((output.lower.sum(1) == 1).all())
            self.assertTrue((output.upper.sum(0) == 1).all())
            self.assertTrue((output.upper.sum(1) == 1).all())

            self.assertTrue(
                np.logical_or(output.lower == 0., output.lower == 1.).all())
            self.assertTrue(
                np.logical_or(output.upper == 0., output.upper == 1.).all())


if __name__ == '__main__':
    main(verbosity=2)
