import numpy


def _delete_weights_in_layer_by_fraction(layer, fraction,
                                         importance_indicator):
    if fraction == 0:
        return

    try:
        W = layer.W
        percentile = numpy.percentile(importance_indicator,
                                      (1 - fraction) * 100)
        W[importance_indicator >= percentile] = 0
        layer.W = W
    except AttributeError:
        pass


def delete_weights_by_layer_fractions(layers, fractions,
                                      importance_indicators):
    """
    Change weights in layer to zeros.

    This function, for every layer,
    changes the given fraction of weights to zeros or,
    if that is not possible, takes the ceiling of such number.
    Weights to be changed are those with bigger indicators.

    :param WeightedLayer layers: layers for sparsifying
    :param float or list or numpy.ndarray fractions:
        fraction of weights to be changed to zeros
    :param numpy.ndarray importance_indicators:
        indicators for each weight for deleting
    """

    try:
        iter(fractions)
    except TypeError:
        fractions = [fractions for _ in layers]

    for layer, fraction, importance_indicator \
            in zip(layers, fractions, importance_indicators):
        _delete_weights_in_layer_by_fraction(layer, fraction,
                                             importance_indicator)


def delete_weights_by_global_fraction(layers, fraction,
                                      importance_indicators):
    """
    Change weights in network to zeros.

    This function, for all layers at once,
    change the given fraction of weights to zeros or,
    if that is not possible, takes the ceiling of such number.
    Weights to be changed are those with bigger indicators.

    If some weight indicator is lower than 0, weight will not be changed to 0.

    :param WeightedLayer layers: layers for sparsifying
    :param float fraction: fraction of weights to be changed to zeros
    :param numpy.ndarray importance_indicators:
        indicators for each weight for deleting
    """

    if fraction == 0:
        return

    flattened_importance_indicators = numpy.concatenate(
        [layer_importance_indicators.flatten()
         for layer_importance_indicators in importance_indicators
         if layer_importance_indicators is not None])

    percentile = numpy.percentile(flattened_importance_indicators,
                                  (1 - fraction) * 100)
    if percentile < 0:
        percentile = 0

    for layer, indicator in zip(layers, importance_indicators):
        try:
            weights = layer.W
            weights[indicator >= percentile] = 0
            layer.W = weights
        except AttributeError:
            pass
