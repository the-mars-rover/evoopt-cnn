# %%file models.py

from tensorflow import keras
from tensorflow.keras import layers


def _get_alexnet(input_shape, num_classes):
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(96, 11, 4, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(3, 2, padding='same'),
            layers.Conv2D(256, 5, 1, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(3, 2, padding='same'),
            layers.Conv2D(384, 3, 1, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(384, 3, 1, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, 1, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(3, 2, padding='same'),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])


def get_model_instantiator(model_name, input_shape, num_classes):
    if model_name == 'alexnet':
        def get_alexnet_instantiator():
            return _get_alexnet(input_shape, num_classes)

        return get_alexnet_instantiator

    return None


__all__ = ['get_model_instantiator']
