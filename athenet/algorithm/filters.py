import numpy as np
from statistics import median

class Filter(object):
    """Filters used to modify convolutional layer's weights
    """
    def apply(self, table):
        """:table: numpy 2D table represents part of convolutional layer's
           weights (one filter). Function modifies elements of the table.
        """
        raise NotImplementedError


def adjust(i, j, shape):
    """adjusts coordinates (i, j) to table shape"""
    if i >= 0 and i < shape[0] and j >= 0 and j < shape[1]:
        return i, j

    if i < 0:
        if j < 0:
            return adjust(i + 1, j + 1, shape)
        return adjust(i + 1, j, shape)
    if j < 0:
        return adjust(i, j + 1, shape)

    if i >= shape[0]:
        if j >= shape[1]:
            return adjust(i - 1, j - 1, shape)
        return adjust(i - 1, j, shape)
    return adjust(i, j - 1, shape)


class WeightedFilter(Filter):
    def __init__(self, W):
        self.W = W

    def apply(self, table):
        result = np.zeros(talbe.shape)
        sum = np.sum(self.W)
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                for fi in xrange(W.shape[0]):
                    for fj in xrange(W.shape[1]):
                        tmpi, tmpj = self.adjust(i - W.shape[0] / 2 + fi,
                                                 j - W.shape[1] / 2 + fj,
                                                 table.shape)
                        result[i][j] += self.W[fi][fj] * table[tmpi][tmpj]
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                table[i][j] = result[i][j]
                if sum != 0:
                    table[i][j] /= sum


avg_filter = WeightedFilter(np.asarray( #removes noise, but blurs image
    [[ 1., 1., 1. ],
     [ 1., 1., 1. ],
     [ 1., 1., 1. ]]
))


lp1_filter = WeightedFilter(np.asarray( #don't blurs image as much as previous
    [[ 1., 1., 1. ],
     [ 1., 2., 1. ],
     [ 1., 1., 1. ]]
))


lp2_filter = WeightedFilter(np.asarray( #don't blurs image as much as previous
    [[ 1., 1., 1. ],
     [ 1., 4., 1. ],
     [ 1., 1., 1. ]]
))


lp3_filter = WeightedFilter(np.asarray( #don't blurs image as much as previous
    [[ 1., 1., 1. ],
     [ 1., 12., 1. ],
     [ 1., 1., 1. ]]
))


hp3_filter = WeightedFilter(np.asarray( #sharpens an image
    [[  0., -1.,  0. ],
     [ -1., 20., -1. ],
     [  0., -1.,  0. ]]
))


vertical_edge_filter = WeightedFilter(np.asarray( #lefts only vertical edges
    [[  0., 0., 0. ],
     [ -1., 1., 0. ],
     [  0., 0., 0. ]]
))


horizontal_edge_filter = WeightedFilter(np.asarray( #lefts only horizontal edges
    [[ 0., -1., 0. ],
     [ 0.,  1., 0. ],
     [ 0.,  0., 0. ]]
))


diagonal_edge_filter_ur = WeightedFilter(np.asarray( #ur means up-right, /
    [[ -1., 0., 0. ],
     [  0., 1., 0. ],
     [  0., 0., 0. ]]
))


diagonal_edge_filter_ul = WeightedFilter(np.asarray( #ul means up-left, \
    [[  0., 0., 0. ],
     [  0., 1., 0. ],
     [ -1., 0., 0. ]]
))


laplace1_filter = WeightedFilter(np.asarray( #lefts only edges (all directions)
    [[  0., -1.,  0. ],
     [ -1.,  4., -1. ],
     [  0., -1.,  0. ]]
))


laplace2_filter = WeightedFilter(np.asarray( #lefts only edges (all directions)
    [[ -1., -1., -1. ],
     [ -1.,  8., -1. ],
     [ -1., -1., -1. ]]
))


laplace3_filter = WeightedFilter(np.asarray( #lefts only edges (all directions)
    [[  1., -2.,  1. ],
     [ -2.,  4., -2. ],
     [  1., -2.,  1. ]]
))


class MedianFilter(Filter):
    def __init__(self, size=3):
        """:size: is fiter's edge size in pixels (integer)"""
        self.size = size

    def apply(self, table):
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[0]):
                weights = []
                for fi in xrange(size):
                    for fj in xrange(size):
                        tmpi, tmpj = i - self.size / 2 + fi,
                                     j - self.size / 2 + fj
                        if tmpi >= 0 and tmpi < table.shape[0] and
                                tmpj >= 0 and tmpj < table.shape[1]:
                            weights.append(table[tmpi][tmpj])
                result[i][j] = median(weights)


#removes noises and not blurs image as much as lp filters
median3_filter = MedianFilter()


median5_filter = MedianFilter(5)