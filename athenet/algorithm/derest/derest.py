"""Functions counting cost of weights in view of their activation/derivative.
"""
import numpy
import theano

from athenet.algorithm.deleting import delete_weights_by_global_fraction
from athenet.algorithm.derest.network import DerestNetwork
from athenet.algorithm.derest.utils import change_order
from athenet.algorithm.numlike.interval import Interval
from athenet.algorithm.utils import to_indicators


def sum_max(values):
    """
    Computes indicator from Numlike values

    :param Numlike values: values to count indicator from
    :return: int
    """
    return values.sum()


def get_derest_indicators(network, input, count_function=sum_max):
    """
    Returns indicators of importance using derest algorithm

    :param Network network: network to work with
    :param Numlike input: possible input for network
    :param function count_function: function to use
    :return array of integers:
    """
    n = DerestNetwork(network)
    n.count_activations(input, True)
    output_nr = network.layers[-1].output_shape
    n.count_derivatives(input.derest_output(output_nr), True)
    results = n.count_derest(count_function)
    return to_indicators(results)


def derest(network, fraction, (min_value, max_value)=(0., 255.)):
    """
    Delete set percentage of weights from network,

    :param Network network: network to delete weights from
    :param float fraction: fraction of weights to be deleted
    :param tuple(float, float) (min_value, max_value):
        range of possible values on input of network
    """

    input_shape = change_order(network.layers[0].input_shape)
    input = Interval(
        theano.shared(numpy.full(input_shape, min_value)),
        theano.shared(numpy.full(input_shape, max_value))
    )
    indicators = get_derest_indicators(network, input, sum_max)
    delete_weights_by_global_fraction(network.weighted_layers,
                                      fraction, indicators)
