"""Auxiliary functions."""

import cPickle as pickle
import os
import sys
import gzip
import urllib
import numpy

from athenet.utils import BIN_DIR, DATA_DIR


def load_data_from_pickle(filename):
    """Load data from pickle file.

    :filename: File with pickled data, may be gzipped.
    :return: Data loaded from file.
    """
    try:
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
    except:
        f = open(filename, 'rb')
        data = pickle.load(f)
    f.close()
    return data


def save_data_to_pickle(data, filename):
    """Saves data to gzipped pickle file.

    :data: Data to be saved.
    :filename: Name of file to save data.
    """
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename, url=None):
    """Load data from file, download file if it doesn't exist.

    :filename: File with pickled data, may be gzipped.
    :url: Url for downloading file.
    :return: Unpickled data.
    """
    if not os.path.isfile(filename):
        if not url:
            return None
        download_file(filename, url)

    data = load_data_from_pickle(filename)
    return data


def download_file(filename, url):
    """Download file from given url.

    :filename: Name of a file to be downloaded.
    :url: Url for downloading file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print 'Downloading ' + os.path.basename(filename) + '...',
    sys.stdout.flush()
    urllib.urlretrieve(url, filename)
    print 'Done'


def get_data_path(name):
    """Return absolute path to the data file.

    :name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(DATA_DIR, name)


def get_bin_path(name):
    """Return absolute path to the binary data file.

    :name: Name of the file.
    :return: Full path to the file.
    """
    return os.path.join(BIN_DIR, name)


def zero_fraction(network):
    """Returns fraction of zeros in weights of Network.

    Biases are not considered.

    :network: Network for which we count fraction of zeros.
    :return: Fraction of zeros.
    """
    params = [layer.W for layer in network.weighted_layers]
    n_non_zero = 0
    n_fields = 0
    for param in params:
        n_fields += numpy.size(param)
        n_non_zero += numpy.count_nonzero(param)
    n_zero = n_fields - n_non_zero
    return (1.0 * n_zero) / (1.0 * n_fields)
