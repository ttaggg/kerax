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


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss."""
    error = y_true - y_pred
    cond = K.abs(error) < clip_delta
    squared_loss = 0.5 * K.square(error)
    linear_loss = clip_delta * (K.abs(error) - 0.5 * clip_delta)
    return K.mean(K.switch(cond, squared_loss, linear_loss))


def l1_l2_loss(y_true, y_pred):
    """MSE + MAE."""
    return (mean_absolute_error(y_true, y_pred) +
            mean_squared_error(y_true, y_pred)) / 2.


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error loss."""
    return K.mean(K.abs(y_pred - y_true))


def mean_squared_error(y_true, y_pred):
    """Mean squared error loss."""
    return K.mean(K.square(y_pred - y_true))
