"""
    Runs chosen algorithm on chosen type of network and prints results.
    With -p flag displays results on the plot.
    By default runs simple_neuron_deleter algorithm on LeNet network on one
    configuration case.

    More information with -h option.
"""


import argparse
import sys
from argparse import RawTextHelpFormatter
from datetime import datetime

from config.algorithm import get_network, ok, deleting, choose_layers,\
    get_indicators
from athenet.utils import run_algorithm, plot_2d_results

parser = argparse.ArgumentParser(
    formatter_class=RawTextHelpFormatter,
    description="Runs chosen indicators on chosen type of network,"
                "delete weights and prints results.")

parser.add_argument("-i", "--indicators",
                    help="Chooses method of computing indicators:\n"
                    " * smallest: get_smallest_indicators (default)\n"
                    " * global_mean: get_nearest_to_global_mean_indicators\n"
                    " * layers_mean: get_nearest_to_layers_mean_indicators\n"
                    " * filters: get_filters_indicators\n"
                    " * derest: get_derest_indicators",
                    choices=["smallest", "global_mean", "layers_mean",
                             "filters", "derest"],
                    default="smallest")

parser.add_argument("-d", "--deleting",
                    help="Chooses way of deleting weights:\n"
                    " * global: delete_weights_by_global_fraction (default)\n"
                    " * layers: delete_weights_by_layer_fraction",
                    choices=["global", "layers"],
                    default="global")

parser.add_argument("-n", "--network",
                    help="Algorithm will be ran on chosen kind of network. "
                    "Default option is \"lenet\".",
                    choices=["lenet", "alexnet", "googlenet"],
                    default="lenet")

parser.add_argument("-t", "--types",
                    help="Chooses type of layers to run algorithm on: \n"
                    " * default: default type of layers for indicators\n"
                    " * all: all layers\n"
                    " * conv: convolution layers\n"
                    " * fully-connected: fully-connected layers\n",
                    choices=["default", "all", "conv", "fully-connected"],
                    default="default")

parser.add_argument("-p", "--plot",
                    help="When this option is added results will be displayed"
                    " on the plot.",
                    action="store_true")

parser.add_argument("-l", "--log",
                    help="When this option is added the plot (if chosed) will"
                    " be displayed on logaritmic scale",
                    action="store_true")

parser.add_argument("-e", "--examples", type=int,
                    help="Chooses how many different fractions of weights "
                         "will be deleted (the higher the number,"
                         "the more thorough the plot "
                         "and the more time it takes)."
                         "Default is 4",
                    default=4)

parser.add_argument("-f", "--file", type=str,
                    help="Name of file to save results to", default=None)


args = parser.parse_args()


print "loading network..."
network = get_network(args.network)
ok()

print "generating indicators..."
ind = get_indicators(network, args.types, args.indicators)
ok()


def deleting_with_indicators(n, p):
    return deleting[args.deleting](
        choose_layers(n, args.types, args.indicators), p, ind)


print "generating results..."
examples = [float(x) / args.examples for x in range(args.examples)]
file_name = args.file if args.file \
    else args.network + "_" + datetime.now().strftime("%d%b_%H:%M:%S:%f")
results = run_algorithm(network, deleting_with_indicators,
                        examples, verbose=True,
                        results_pkl=file_name).get_zeros_fraction()
ok()

if args.plot:
    plot_2d_results(results, ylog=args.log,
                    title="results of " + args.indicators + " indicators"
                    " deleted by " + args.deleting + " fraction")
