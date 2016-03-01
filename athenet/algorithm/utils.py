import numpy as np


def list_of_percentage_rows_table(table, layer_id):
    """returns list of tuples (percentage, number of row, [layer_id])
       representing rows of [table]

       "percentage" is a sum of absolute values in row divided by
       sum of absolute values in all table
    """
    sum_of_all = np.sum(abs(table))
    result = []
    for i in xrange(table.shape[0]):
        result.append((np.sum(abs(table[i])) / sum_of_all, i, layer_id))
    return result


def list_of_percentage_columns(layer_id, layer):
    return list_of_percentage_rows_table(np.transpose(layer.W), layer_id)


def list_of_percentage_rows(layer_id, layer):
    return list_of_percentage_rows_table(layer.W, layer_id)


def delete_column(layer, i):
    W = layer.W
    for j in xrange(W.shape[0]):
        W[j][i] = 0.
    layer.W = W


def delete_row(layer, i):
    W = layer.W
    for j in xrange(W.shape[1]):
        W[i][j] = 0.
    layer.W = W
