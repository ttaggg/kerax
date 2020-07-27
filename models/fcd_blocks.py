"""Blocks used for FC-Densenet in functional style."""

from keras import layers
from keras import regularizers


class ConvoBlock:
    """Convolution block."""

    def __init__(self, filters_size, dropout_rate):
        super().__init__()
        self._batchnorm = layers.BatchNormalization()
        self._relu = layers.Activation('relu')
        self._convolution = layers.Conv2D(
            filters_size,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4))
        self._dropout = layers.Dropout(rate=dropout_rate)

    def __call__(self, inputs, is_training):
        # NOTE(ttaggg): authors used batch normalization
        # in training mode both during training and inference.
        outputs = self._batchnorm(inputs)
        outputs = self._relu(outputs)
        outputs = self._convolution(outputs)
        outputs = self._dropout(outputs, training=is_training)
        return outputs


class DenseBlock:
    """Dense block."""

    def __init__(self, filters_size, dropout_rate, n_layers):
        self._concat_layers = [
            layers.Concatenate(axis=-1) for _ in range(n_layers)
        ]
        self._convo_blocks = [
            ConvoBlock(filters_size, dropout_rate) for _ in range(n_layers)
        ]

    def __call__(self, inputs, is_training):
        stack = inputs
        blocks = []
        for convo_block, concat in zip(self._convo_blocks, self._concat_layers):
            convo = convo_block(stack, is_training)
            stack = concat([stack, convo])
            blocks.append(convo)
        return stack, blocks


class DownBlock:
    """Downsampling block."""

    def __init__(self, filters_size, dropout_rate):
        self._convo_block = ConvoBlock(filters_size, dropout_rate)
        self._max_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))

    def __call__(self, inputs, is_training):
        outputs = self._convo_block(inputs, is_training)
        outputs = self._max_pool(outputs)
        return outputs


class UpBlock:
    """Upsampling block."""

    def __init__(self, filters_size):
        self._concatenation = layers.Concatenate(axis=-1)
        self._convolution_transpose = layers.Conv2DTranspose(
            filters_size,
            kernel_size=(3, 3),
            padding='same',
            strides=(2, 2),
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4))

    def __call__(self, inputs):
        outputs = self._concatenation(inputs)
        outputs = self._convolution_transpose(outputs)
        return outputs
