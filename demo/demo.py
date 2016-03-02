import argparse
from copy import deepcopy
from consts import datasets, algorithms, get_network, ok
from athenet.utils import run_algorithm, plot_2d_results
from athenet.utils import MNISTDataLoader

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--algorithm",
                   help="algorithm which result be demonstrated",
                   choices=["sender", "sender2"],
                   default="sender")
parser.add_argument("-n", "--network",
                   help="neural network given to run the algorithm on",
                   choices=["lenet"],
                   default="lenet")
parser.add_argument("-p", "--plot",
                   help="Displeys results on the plot.",
                   action="store_true")
parser.add_argument("-d", "--dataset", type=int,
                    help="Number of dataset. Datasets are numbered from 0."
                    " (default value). Amount of datasets depends on algorithm"
                    "(sender: 2,"
                    " sender2: 2)",
                    default=0)

args = parser.parse_args()

dataset = datasets[args.algorithm][args.dataset]
algorithm = algorithms[args.algorithm]
print "loading network..."
network = get_network(args.network)
ok()

print "generating results..."
results = run_algorithm(network, algorithm, dataset)
ok()
for config in dataset:
    print "for config", config
    print "zeroed_fraction:", results[config][0]
    print "error rate:", results[config][1]

if args.plot:
    plot_2d_results(results)

