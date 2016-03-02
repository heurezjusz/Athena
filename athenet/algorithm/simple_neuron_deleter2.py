"""
    sender - SimplE Neuron DEleteR

    Deletes the least significant neurons.
    Outgoing value of neuron is defined as sum of absolute values of
    weights outgoing from neuron divided by sum of absolute values of
    weights outgoing from entire layer.
    Ingoing value of neuron is defined as sum of absolute values of
    weights ingoing from neuron divided by sum of absolute values of
    weights ingoing from entire layer.
    Neuron value is defined as multiplication of its ingoing and outgoing
    values.
"""

import numpy as np

from athenet.layers import FullyConnectedLayer, ConvolutionalLayer, MaxPool
from athenet.algorithm.utils import list_of_percentage_columns, \
    list_of_percentage_rows, delete_column, delete_row


def simple_neuron_deleter2(network, config):
    """
        :network: - an instance of athenet.Network.
        :config: - tuple of 2 foats, p and layer_limit
        :p, layer_limit: - floats between 0 and 1.

        Modifies [network]. Deletes [p] neurons from layers connected direclty
        to fully connected layer's. Do not delete more than [layer_limit]
        neurons from single layer.
        If [layer_limit] < [p] then at most [layer_limit] neurons will be
        deleted.

        Deletion of neuron is simulated by setting all weights outgoing
        form to it and ingoing to it to 0. In athenet.network they are
        reprezented as rows of next layer's weights matrix.

        The algorithm supposes that there is no ConvolutionalLayer or MaxPool
        layer between any FullyConnectedLayers
    """
    p, layer_limit = config

    assert p >= 0. and p <= 1.
    assert layer_limit >= 0. and layer_limit <= 1.
    if layer_limit < p:
        p = layer_limit

    # counter of neurons
    neurons_for_layer = np.zeros((len(network.layers)))
    neurons_in_general = 0
    # counter of deleted neurons
    deleted_for_layer = np.zeros((len(network.layers)))
    deleted_in_general = 0

    # weights is list of tuples (outgoing values of provious layer (rows),
    # ingoing values of this layer (columns))
    # ingoing / outgoing value = (value, number of row / column, layer_id)
    weights = []
    was_fully_connected_layer = False
    for i in xrange(len(network.layers)):
        layer = network.layers[i]
        if isinstance(layer, FullyConnectedLayer):
            was_fully_connected_layer = True
            weights.append((list_of_percentage_rows(i, layer),
                            list_of_percentage_columns(i, layer)))
            neurons_for_layer[i] = layer.W.shape[1]
            neurons_in_general += neurons_for_layer[i]
        elif was_fully_connected_layer:
            assert not isinstance(layer, ConvolutionalLayer)
            assert not isinstance(layer, MaxPool)

    considered_neurons = []
    for i in xrange(len(weights) - 1):
        for j in xrange(len(weights[i][1])):
            # neuron is reprezented as tuple
            #(value, number of column (and row), column_layer_id, row_layer_id)
            assert weights[i][1][j][1] == weights[i + 1][0][j][1]
            considered_neurons.append(
                (weights[i][1][j][0] * weights[i + 1][0][j][0],
                 weights[i][1][j][1],
                 weights[i][1][j][2],
                 weights[i + 1][0][j][2])
            )

    considered_neurons = sorted(considered_neurons)
    for val, neuron_id, column_layer_id, row_layer_id in considered_neurons:
        if deleted_in_general >= p * neurons_in_general:
            break
        if deleted_for_layer[row_layer_id] + 1 > \
                layer_limit * neurons_for_layer[row_layer_id]:
            continue
        delete_column(network.layers[column_layer_id], neuron_id)
        delete_row(network.layers[row_layer_id], neuron_id)
        deleted_for_layer[row_layer_id] += 1
        deleted_in_general += 1
