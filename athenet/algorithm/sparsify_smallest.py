import numpy

from athenet.algorithm.deleting import delete_weights_by_layer_fractions,\
    delete_weights_by_global_fraction
from athenet.algorithm.utils import to_indicators


def get_smallest_indicators(layers):
    """
    Return indicators of smallest weights in layers.

    This function, for a given set of layers,
    computes importance indicators for every weight
    based on how big it is.
    Smaller weights will be marked as more likely to delete.

    :param iterable layers: layers to get indicators for
    :return: indicators
    """
    return to_indicators([layer.W for layer in layers])


def get_nearest_to_global_mean_indicators(layers):
    """
    Return indicators of weights in layers nearest
    to global weight mean.

    This function, for a given set of layers, computes global weights mean
    and returns importance indicators for every weight
    based on how close to global mean it is.
    Weights closest to global mean will be marked as more likely to delete.

    :param iterable layers: layers to get indicators for
    :return: indicators
    """

    weights = numpy.concatenate(
        [layer.W.flatten() for layer in layers])
    mean = numpy.mean(weights)
    return to_indicators([abs(mean - layer.W) for layer in layers])


def _get_nearest_to_layer_mean_indicators(layer):
    mean = numpy.mean(layer.W)
    return to_indicators(mean - layer.W)


def get_nearest_to_layers_mean_indicators(layers):
    """
    Return indicators of weights in layers nearest
    to layer weight mean.

    This function, for every given layer, computes weights mean
    and returns importance indicators for every weight
    based on how close to it's layer mean it is.
    Weights closest to it's layer mean will be marked as more likely to delete.

    :param iterable layers: layers to get indicators for
    :return: indicators
    """

    return numpy.array([_get_nearest_to_layer_mean_indicators(layer)
                        for layer in layers])


def sparsify_smallest_on_network(network, zeroed_weights_fraction):
    """
    Change smallest weights in network to zeros.

    This function changes weights in such a way,
    that for the whole network the given fraction
    of the smallest of them are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        percentage of weights to be changed to zeros
    """

    indicators = get_smallest_indicators(network.weighted_layers)
    delete_weights_by_global_fraction(network.weighted_layers,
                                      zeroed_weights_fraction, indicators)


def sparsify_nearest_to_network_mean(network, zeroed_weights_fraction):
    """
    Change weights close to network's mean to zeros.

    This function changes weights in such a way,
    that for the whole network the given fraction
    of the closest of them to mean are zeros,
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changes to zeros
    """

    indicators = get_nearest_to_global_mean_indicators(
        network.weighted_layers)
    delete_weights_by_global_fraction(network.weighted_layers,
                                      zeroed_weights_fraction, indicators)


def sparsify_smallest_on_layers(network, zeroed_weights_fraction):
    """
    Change smallest weights in each layer to zeros.

    This function for each layer changes weights in such a way,
    that at least the given fraction of the smallest of them are zeros.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    """

    layers = network.weighted_layers
    indicators = get_smallest_indicators(layers)
    delete_weights_by_layer_fractions(layers, zeroed_weights_fraction,
                                      indicators)


def sparsify_nearest_to_layer_mean(network, zeroed_weights_fraction):
    """
    In each layer, change weights closest to mean to zeros.

    This function for each layer changes weights in such a way,
    that the given fraction of the closest to mean are set to zero
    or, if that is not possible, takes the ceiling of such a number.

    :param Network network: network for sparsifying
    :param float zeroed_weights_fraction:
        fraction of weights to be changed to zeros
    """

    layers = network.weighted_layers
    indicators = get_nearest_to_layers_mean_indicators(layers)
    delete_weights_by_layer_fractions(layers, zeroed_weights_fraction,
                                      indicators)
