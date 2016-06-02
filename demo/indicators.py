"""
    Runs chosen algorithm on chosen type of network and prints results.
    With -p flag displays results on the plot.
    By default runs simple_neuron_deleter algorithm on LeNet network on one
    configuration case.

    More information with -h option.
"""


import argparse
from argparse import RawTextHelpFormatter

from config.algorithm import get_network, ok, deleting, choose_layers,\
    get_indicators, get_file_name
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

parser.add_argument("-b", "--batch_size", type=int,
                    help="Chooses maximum size of batches uses to compute"
                         "derivatives in derest algoritm"
                         "(values lower than 1 means there will be "
                         "no maximum size)",
                    default=0)

parser.add_argument("-f", "--file", type=str,
                    help="Name of file to save results to", default=None)


args = parser.parse_args()


print "loading network..."
network = get_network(args.network)
ok()

print "generating indicators..."
ind = get_indicators(network, args.types, args.indicators, args.batch_size)
ok()


def deleting_with_indicators(n, p):
    return deleting[args.deleting](
        choose_layers(n, args.types, args.indicators), p, ind)


print "generating results..."
n = args.examples
if n > 1:
    examples = [x / (n - 1.) for x in range(n)]
elif n == 1:
    examples = [0.5]
else:
    examples = []

file_name = get_file_name(args.file, args.network)

results = run_algorithm(network, deleting_with_indicators,
                        examples, verbose=True,
                        results_pkl=file_name).get_zeros_fraction()
ok()

if args.plot:
    # TODO: Make filename
    plot_2d_results(results, 'dummy_filename_todo', ylog=args.log,
                    title="results of " + args.indicators + " indicators"
                    " deleted by " + args.deleting + " fraction")
