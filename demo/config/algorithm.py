from datetime import datetime

from athenet.algorithm import sparsify_smallest_on_network, sharpen_filters, \
    sparsify_smallest_on_layers, simple_neuron_deleter,\
    simple_neuron_deleter2, derest
from athenet.algorithm import get_filters_indicators, get_smallest_indicators,\
    get_nearest_to_global_mean_indicators, \
    get_nearest_to_layers_mean_indicators, get_derest_indicators, \
    delete_weights_by_global_fraction, delete_weights_by_layer_fractions
from athenet.models import lenet, alexnet, googlenet
from athenet.data_loader import MNISTDataLoader, ImageNetDataLoader, \
    AlexNetImageNetDataLoader, GoogleNetImageNetDataLoader
from athenet.layers import FullyConnectedLayer, ConvolutionalLayer


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
                (0.8, 1.0), (0.9, 1.0)],
               [(0.02, 0.75), (0.04, 0.75), (0.06, 0.75), (0.08, 0.75),
                (0.1, 0.75), (0.12, 0.75), (0.14, 0.75), (0.16, 0.75),
                (0.18, 0.75), (0.2, 0.75), (0.22, 0.75), (0.24, 0.75),
                (0.26, 0.75), (0.28, 0.75), (0.3, 0.75), (0.325, 0.75),
                (0.35, 0.75), (0.375, 0.75), (0.4, 0.75), (0.45, 0.75),
                (0.5, 0.75), (0.55, 0.75), (0.6, 0.75), (0.7, 0.75),
                (0.8, 0.75), (0.9, 0.75)],
               [(0.02, 0.5), (0.04, 0.5), (0.06, 0.5), (0.08, 0.5), (0.1, 0.5),
                (0.12, 0.5), (0.14, 0.5), (0.16, 0.5), (0.18, 0.5), (0.2, 0.5),
                (0.22, 0.5), (0.24, 0.5), (0.26, 0.5), (0.28, 0.5), (0.3, 0.5),
                (0.325, 0.5), (0.35, 0.5), (0.375, 0.5), (0.4, 0.5),
                (0.45, 0.5), (0.5, 0.5), (0.55, 0.5), (0.6, 0.5), (0.7, 0.5),
                (0.8, 0.5), (0.9, 0.5)]
               ],
    "sender2": [[(0.3, 0.75)],
                [(0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.08, 1.0),
                 (0.1, 1.0), (0.12, 1.0), (0.14, 1.0), (0.16, 1.0),
                 (0.18, 1.0), (0.2, 1.0), (0.22, 1.0), (0.24, 1.0),
                 (0.26, 1.0), (0.28, 1.0), (0.3, 1.0), (0.325, 1.0),
                 (0.35, 1.0), (0.375, 1.0), (0.4, 1.0), (0.45, 1.0),
                 (0.5, 1.0), (0.55, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0),
                 (0.9, 1.0)],
                [(0.02, 0.75), (0.04, 0.75), (0.06, 0.75), (0.08, 0.75),
                 (0.1, 0.75), (0.12, 0.75), (0.14, 0.75), (0.16, 0.75),
                 (0.18, 0.75), (0.2, 0.75), (0.22, 0.75), (0.24, 0.75),
                 (0.26, 0.75), (0.28, 0.75), (0.3, 0.75), (0.325, 0.75),
                 (0.35, 0.75), (0.375, 0.75), (0.4, 0.75), (0.45, 0.75),
                 (0.5, 0.75), (0.55, 0.75), (0.6, 0.75), (0.7, 0.75),
                 (0.8, 0.75), (0.9, 0.75)],
                [(0.02, 0.5), (0.04, 0.5), (0.06, 0.5), (0.08, 0.5),
                 (0.1, 0.5), (0.12, 0.5), (0.14, 0.5), (0.16, 0.5),
                 (0.18, 0.5), (0.2, 0.5), (0.22, 0.5), (0.24, 0.5),
                 (0.26, 0.5), (0.28, 0.5), (0.3, 0.5), (0.325, 0.5),
                 (0.35, 0.5), (0.375, 0.5), (0.4, 0.5), (0.45, 0.5),
                 (0.5, 0.5), (0.55, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5),
                 (0.9, 0.5)],
                [(x / 100., 1.) for x in xrange(1, 21)]],
    "rat":     [[0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                 0.22, 0.24, 0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45,
                 0.5, 0.55, 0.6, 0.7, 0.8, 0.9],
                [x / 50. for x in xrange(1, 30)] +
                [x / 40. for x in xrange(24, 40)],
                [x / 100. for x in xrange(1, 21)]],
    "rat2":     [[0.5],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                 [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                  0.22, 0.24, 0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45,
                  0.5, 0.55, 0.6, 0.7, 0.8, 0.9],
                 [x / 50. for x in xrange(1, 30)] +
                 [x / 40. for x in xrange(24, 40)]],
    "filters": [[(0.3, 1, (5, 75, 75))],
                [(x / 10., 1, (5, 75, 75)) for x in xrange(1, 10)],
                [(x / 20., 1, (5, 75, 75)) for x in xrange(1, 10)]],
    "derest": [[0.5],
               [(x / 10.) for x in xrange(1, 10)],
               [(x / 30.) for x in xrange(1, 20)]]
    }


"""
    dictionary form algorithm shortcut to function to be called
"""
algorithms = {
    "sender": simple_neuron_deleter,
    "sender2": simple_neuron_deleter2,
    "rat": sparsify_smallest_on_network,
    "rat2": sparsify_smallest_on_layers,
    "filters": sharpen_filters,
    "derest": derest
    }


indicators = {
    "smallest": get_smallest_indicators,
    "global_mean": get_nearest_to_global_mean_indicators,
    "layers_mean": get_nearest_to_layers_mean_indicators,
    "filters": get_filters_indicators,
    "derest": get_derest_indicators
}

default_types_of_layers = {
    "smallest": "all",
    "global_mean": "all",
    "layers_mean": "all",
    "filters": "conv",
    "derest": "all"
}

deleting = {
    "global": delete_weights_by_global_fraction,
    "layers": delete_weights_by_layer_fractions
}


def choose_layers(network, type_, indicators_, from_=None):
    if from_ is None:
        from_ = network.weighted_layers

    def get_layers(network, type_, indicators_):
        if type_ == "default":
            return get_layers(network, default_types_of_layers[indicators_],
                              indicators_)
        if type_ == "all":
            return [True for _ in network.weighted_layers]
        if type_ == "conv":
            return [isinstance(layer, ConvolutionalLayer)
                    for layer in network.weighted_layers]
        if type_ == "fully-connected":
            return [isinstance(layer, FullyConnectedLayer)
                    for layer in network.weighted_layers]

    layers = get_layers(network, type_, indicators_)
    return [a for (a, b) in zip(from_, layers) if b]


def get_indicators(network, type_, indicators_, batch_size):
    f = indicators[indicators_]
    if indicators_ == "derest":
        max_batch_size = batch_size if batch_size > 0 else None
        ind = f(network, max_batch_size=max_batch_size)
        return choose_layers(network, type_, indicators_, ind)
    else:
        return f(choose_layers(network, type_, indicators_))


def get_network(network_type):
    """
        Returns a athenet.network of given type.
        :network_type: is a name of the type given as a string.
    """
    if network_type == "lenet":
        net = lenet()
        net.data_loader = MNISTDataLoader()
        return net
    if network_type == "alexnet":
        net = alexnet()
        net.data_loader = AlexNetImageNetDataLoader()
        return net
    if network_type == "googlenet":
        net = googlenet()
        net.data_loader = GoogleNetImageNetDataLoader()
        return net
    raise NotImplementedError

def get_file_name(proposed_file_name, network_type):
    return proposed_file_name if proposed_file_name is not None else \
        network_type + "_" + datetime.now().strftime("%d_%b_%H_%M_%S_%f")


def ok():
    """
        prints a good looking "OK" on the stdout.
    """
    print "[ \033[32mOK\033[39m ]"
