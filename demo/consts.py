from athenet.algorithm import simple_neuron_deleter, simple_neuron_deleter2
from athenet.models import lenet
from athenet.utils import MNISTDataLoader


datasets = {
    "sender": [[(0.3, 0.75)],
               [(0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.08, 1.0), (0.1, 1.0),
                (0.12, 1.0), (0.14, 1.0), (0.16, 1.0), (0.18, 1.0), (0.2, 1.0),
                (0.22, 1.0), (0.24, 1.0), (0.26, 1.0), (0.28, 1.0), (0.3, 1.0),
                (0.325, 1.0), (0.35, 1.0), (0.375, 1.0), (0.4, 1.0),
                (0.45, 1.0), (0.5, 1.0), (0.55, 1.0), (0.6, 1.0), (0.7, 1.0),
                (0.8, 1.0), (0.9, 1.0)]],
    "sender2": [[(0.3, 0.75)],
               [(0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.08, 1.0), (0.1, 1.0),
                (0.12, 1.0), (0.14, 1.0), (0.16, 1.0), (0.18, 1.0), (0.2, 1.0),
                (0.22, 1.0), (0.24, 1.0), (0.26, 1.0), (0.28, 1.0), (0.3, 1.0),
                (0.325, 1.0), (0.35, 1.0), (0.375, 1.0), (0.4, 1.0),
                (0.45, 1.0), (0.5, 1.0), (0.55, 1.0), (0.6, 1.0), (0.7, 1.0),
                (0.8, 1.0), (0.9, 1.0)]]
    }


algorithms = {
    "sender": simple_neuron_deleter,
    "sender2": simple_neuron_deleter2
    }


def get_network(network_type):
    if network_type == "lenet":
        net = lenet()
        net.data_loader = MNISTDataLoader()
        return net
    raise NotImplementedError


def ok():
    print "[ \033[32mOK\033[39m ]"