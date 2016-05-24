from athenet.algorithm.derest.derest import derest
from athenet.network import Network
from athenet.layers import ConvolutionalLayer, FullyConnectedLayer,\
    Softmax, ReLU, MaxPool


n = Network([
        ConvolutionalLayer(image_shape=(10, 10, 1), filter_shape=(4, 4, 2)),
        ReLU(),
        MaxPool(poolsize=(2, 2)),
        ConvolutionalLayer(filter_shape=(2, 2, 5)),
        ReLU(),
        MaxPool(poolsize=(2, 2)),
        FullyConnectedLayer(n_out=10),
        ReLU(),
        FullyConnectedLayer(n_out=3),
        Softmax(),
])

derest(n, 0.6)