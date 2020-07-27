"""Loss functions."""

import keras.backend as K


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy."""
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def binary_cross_entropy_dice(y_true, y_pred):
    """Binary cross-entropy with Dice."""
    return binary_cross_entropy(y_true, y_pred) + dice(y_true, y_pred)


def cross_entropy(y_true, y_pred):
    """Categorial cross-entropy."""
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def cross_entropy_dice(y_true, y_pred):
    """Cross-entropy with Dice loss."""
    return cross_entropy(y_true, y_pred) + dice(y_true, y_pred)


def dice(y_true, y_pred, smooth=1.0):
    """Dice loss."""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice_coeff = (2. * intersection + smooth) / (
        K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - dice_coeff
