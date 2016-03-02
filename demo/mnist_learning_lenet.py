"""Training LeNet on MNIST data."""

import athenet
from athenet.data_loader import MNISTDataLoader


network = athenet.models.lenet(trained=False)
network.data_loader = MNISTDataLoader()

network.train(learning_rate=0.1, n_epochs=10, batch_size=300)
print 'Accuracy on test data: {:.2f}%'.format(100*network.test_accuracy())
