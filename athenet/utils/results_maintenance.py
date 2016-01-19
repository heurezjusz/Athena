#!/usr/bin/python2

import matplotlib.pyplot as plt
from athenet.utils import save_data_to_pickle, load_data_from_pickle


def merge_pickles(out_pkl, in_pkl1, in_pkl2):
    """ Merge pickled dictionaries.

    Merges 2 pickled dictionaries from two files named in_pkl1 and in_pkl2.
    Pickles results to file named out_pkl.

    :out_pkl: Output pickle with merged
    in_pkl1: Name of file with first pickled dictionary.
    in_pkl2: Name of file with second pickled dictionary.
    """
    pkl1 = load_data_from_pickle(in_pkl1)
    pkl2 = load_data_from_pickle(in_pkl2)
    pkl1.update(pkl2)
    save_data_to_pickle(pkl1, out_pkl)


def plot_2d_results(results, xlabel='fraction of zero-filled weights',
                    ylabel='error rate', xlog=False, ylog=False,
                    title='No title'):
    """Create 2d plot of results given by sparsifying algorithm.

    Creates 2d plot with results' values on axes.

    :results: List of pairs to be plotted or dictionary with pairs as values
              (for any key). Both elements from pair must be recognized by
              matplotlib.pylot as numbers.
    :xlabel: Label for x axis.
    :ylabel: Label for y axis.
    :xlog: If true, x axis is logarithmic.
    :ylog: If true, y axis is logarithmic.
    :title: Title of the graph.
    """
    if type(results) is dict:
        results = results.values()
    zf_fraction_l = [res[0] for res in results]
    error_rate_l = [res[1] for res in results]
    plt.plot(zf_fraction_l, error_rate_l, 'ro')
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
