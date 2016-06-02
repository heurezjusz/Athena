"""Auxiliary functions for managing results of algorithms"""

import matplotlib.pyplot as plt
from athenet.utils import save_data_to_pickle, load_data_from_pickle
from athenet.utils import RESULTS_DIR

def merge_pickles(out_pkl, in_pkl1, in_pkl2):
    """ Merge pickled dictionaries.

    Merges two pickled dictionaries from two files named in_pkl1 and in_pkl2.
    Pickles results to file named out_pkl.

    :param out_pkl: Output pickle with merged dictionaries.
    :param in_pkl1: Name of file with first pickled dictionary.
    :param in_pkl2: Name of file with second pickled dictionary.
    """
    pkl1 = load_data_from_pickle(in_pkl1)
    pkl2 = load_data_from_pickle(in_pkl2)
    pkl1.update(pkl2)
    save_data_to_pickle(pkl1, out_pkl)


def plot_2d_results(results, filename,
                    xlabel='fraction of zero-filled weights',
                    ylabel='error rate', xlog=False, ylog=False, title=None):
    """Create 2d plot of results given by sparsifying algorithm.

    Creates 2d plot with results' values on axes.

    :param results: List of pairs to be plotted or dictionary with pairs as
                    values (for any key). Both elements from pair must be
                    recognized by matplotlib.pylot as numbers.
    :param filename: Name of file where results will be saved. Files are saved
                     in RESULTS_DIR directory. Png extension will be
                     automatically added.
    :param xlabel: Label for x axis.
    :param ylabel: Label for y axis.
    :param xlog: If true, x axis is logarithmic.
    :param ylog: If true, y axis is logarithmic.
    :param title: Title of the graph.
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
    if title:
        plt.title(title)
    os_filename = RESULTS_DIR + filename + '.png'
    plt.savefig(os_filename)
    #plt.show()
