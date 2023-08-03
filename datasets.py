# %%file datasets.py

import numpy
import tensorflow

from tensorflow import keras

def _load_fashion_mnist(batch_size):
    input_shape = (28, 28, 1)
    num_classes = 10
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]

    options = tensorflow.data.Options()
    options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.FILE
    return (
        input_shape, num_classes,
        tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).cache().batch(batch_size).with_options(options),
        tensorflow.data.Dataset.from_tensor_slices((x_val, y_val)).cache().batch(batch_size).with_options(options),
        tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).cache().batch(batch_size).with_options(options),
    )


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'fashion_mnist':
        return _load_fashion_mnist(batch_size)

    return None


__all__ = ['load_dataset']
