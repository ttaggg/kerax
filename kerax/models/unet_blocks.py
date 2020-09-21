"""Blocks used for UNet in functional style."""

from keras import layers
from keras import regularizers


class ConvoBlock:
    """Convolution block."""

    def __init__(self, filters_size):
        super().__init__()
        self._convolution_1 = layers.Conv2D(
            filters_size,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4))
        self._batchnorm_1 = layers.BatchNormalization()
        self._relu_1 = layers.Activation('relu')
        self._convolution_2 = layers.Conv2D(
            filters_size,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4))
        self._batchnorm_2 = layers.BatchNormalization()
        self._relu_2 = layers.Activation('relu')

    def __call__(self, inputs, is_training):
        outputs = self._convolution_1(inputs)
        outputs = self._batchnorm_1(outputs, training=is_training)
        outputs = self._relu_1(outputs)
        outputs = self._convolution_2(outputs)
        outputs = self._batchnorm_2(outputs, training=is_training)
        outputs = self._relu_2(outputs)
        return outputs


class DownBlock:
    """Downsampling block."""

    def __init__(self, filters_size, dropout_rate):
        self._convo_block = ConvoBlock(filters_size)
        self._max_pool = layers.MaxPooling2D((2, 2))
        self._dropout = layers.SpatialDropout2D(rate=dropout_rate)

    def __call__(self, inputs, is_training):
        outputs_nopool = self._convo_block(inputs, is_training)
        outputs = self._max_pool(outputs_nopool)
        outputs = self._dropout(outputs, training=is_training)
        return outputs, outputs_nopool


class UpBlock:
    """Upsampling block."""

    def __init__(self, filters_size, dropout_rate):
        self._convolution_transpose = layers.Conv2DTranspose(
            filters_size,
            kernel_size=(3, 3),
            padding='same',
            strides=(2, 2),
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4))
        self._concatenation = layers.Concatenate(axis=-1)
        self._convo_block = ConvoBlock(filters_size)
        self._dropout = layers.SpatialDropout2D(rate=dropout_rate)

    def __call__(self, inputs, skip, is_training):
        outputs = self._convolution_transpose(inputs)
        outputs = self._concatenation([outputs, skip])
        outputs = self._convo_block(outputs, is_training)
        outputs = self._dropout(outputs, training=is_training)
        return outputs
