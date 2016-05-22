from athenet.algorithm import sparsify_smallest_on_network, sharpen_filters
from athenet.algorithm import simple_neuron_deleter, simple_neuron_deleter2,\
    derest
from athenet.models import lenet
from athenet.data_loader import MNISTDataLoader


"""
    Dataset - set (list) of configs given to algorithm as an input.
    "datasets" is a dictionary from algorithm shortcut to list of available
    datasets.
    Do not forget to update help message after changing!
"""
datasets = {
    "sender": [[(0.3, 0.75)],
               [(0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.08, 1.0), (0.1, 1.0),
                (0.12, 1.0), (0.14, 1.0), (0.16, 1.0), (0.18, 1.0), (0.2, 1.0),
                (0.22, 1.0), (0.24, 1.0), (0.26, 1.0), (0.28, 1.0), (0.3, 1.0),
                (0.325, 1.0), (0.35, 1.0), (0.375, 1.0), (0.4, 1.0),
                (0.45, 1.0), (0.5, 1.0), (0.55, 1.0), (0.6, 1.0), (0.7, 1.0),
                (0.8, 1.0), (0.9, 1.0)]],
    "sender2": [[(0.3, 0.75)],
               [(0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.08, 1.0), (0.1, 1.0),
                (0.12, 1.0), (0.14, 1.0), (0.16, 1.0), (0.18, 1.0), (0.2, 1.0),
                (0.22, 1.0), (0.24, 1.0), (0.26, 1.0), (0.28, 1.0), (0.3, 1.0),
                (0.325, 1.0), (0.35, 1.0), (0.375, 1.0), (0.4, 1.0),
                (0.45, 1.0), (0.5, 1.0), (0.55, 1.0), (0.6, 1.0), (0.7, 1.0),
                (0.8, 1.0), (0.9, 1.0)]],
    "rat":    [[0.5],
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]],
    "filters": [[(0.3, 1, (5, 75, 75))],
                [(x / 10., 1, (5, 75, 75)) for x in xrange(1, 10)],
                [(x / 20., 1, (5, 75, 75)) for x in xrange(1, 10)]],
    "derest": [[0.5],
               [[x / 10.] for x in xrange(1, 10)],
               [[x / 30.] for x in xrange(1, 20)]]
    }


"""
    dictionary form algorithm shortcut to function to be called
"""
algorithms = {
    "sender": simple_neuron_deleter,
    "sender2": simple_neuron_deleter2,
    "rat": sparsify_smallest_on_network,
    "filters": sharpen_filters,
    "derest": derest
    }


def get_network(network_type):
    """
        Returns a athenet.network of given type.
        :network_type: is a name of the type given as a string.
    """
    if network_type == "lenet":
        net = lenet()
        net.data_loader = MNISTDataLoader()
        return net
    raise NotImplementedError


def ok():
    """
        prints a good looking "OK" on the stdout.
    """
    print "[ \033[32mOK\033[39m ]"
