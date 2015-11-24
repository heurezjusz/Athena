"""
Program for loading net from a file.
"""

import cPickle
import LeNet.convolutional_mlp as conv

weights, biases = cPickle.load(open('net.pkl'))
conv.test_lenet5(weights, biases)