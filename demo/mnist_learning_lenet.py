"""Training LeNet on MNIST data."""

from athenet import Network
from athenet.layers import ReLU, Softmax, MaxPool, FullyConnectedLayer,\
    ConvolutionalLayer
from athenet.utils import MNISTDataLoader


network = Network([
    ConvolutionalLayer(image_shape=(28, 28, 1), filter_shape=(5, 5, 20)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    ConvolutionalLayer(filter_shape=(5, 5, 50)),
    ReLU(),
    MaxPool(poolsize=(2, 2)),
    FullyConnectedLayer(n_out=500),
    ReLU(),
    FullyConnectedLayer(n_out=10),
    Softmax(),
])

network.data_loader = MNISTDataLoader()
network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
