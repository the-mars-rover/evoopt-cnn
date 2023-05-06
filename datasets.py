# %%file datasets.py

import numpy

from tensorflow import keras


def _load_fashion_mnist():
    num_classes = 10
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = numpy.expand_dims(x_train, -1)
    x_test = numpy.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (input_shape, num_classes), (x_train, y_train), (x_test, y_test)


def load_dataset(dataset_name):
    if dataset_name == 'fashion_mnist':
        return _load_fashion_mnist()

    return None


__all__ = ['load_dataset']
