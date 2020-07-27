"""Metrics."""

import keras.backend as K


def dice(y_true, y_pred, smooth=1.0):
    """Dice metric."""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    if K.sum(y_true) == 0 and K.sum(y_pred) != 0:
        return 0

    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        K.sum(y_true) + K.sum(y_pred) + smooth)


def mcc(y_true, y_pred):
    """Matthews correlation coefficient."""

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    true_pos = K.sum(y_pos * y_pred_pos)
    true_neg = K.sum(y_neg * y_pred_neg)

    false_pos = K.sum(y_neg * y_pred_pos)
    false_neg = K.sum(y_pos * y_pred_neg)

    numerator = (true_pos * true_neg - false_pos * false_neg)
    denominator = K.sqrt((true_pos + false_pos) * (true_pos + false_neg) *
                         (true_neg + false_pos) * (true_neg + false_neg))

    return numerator / (denominator + K.epsilon())
