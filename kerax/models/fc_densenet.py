"""FC-Densenet
Fully convolutional densenet as in:
https://arxiv.org/pdf/1611.09326v2.pdf
"""
import itertools

import keras.backend as K
from keras import layers
from keras import models
from keras import regularizers

from kerax.models import fcd_blocks as blocks


class TransitionDown:
    """Downsampling transition."""

    def __init__(self, block_lengths, filter_sizes, growth_rate, dropout_rate):

        self._block_lengths = block_lengths

        self._dense_blocks = [
            blocks.DenseBlock(growth_rate, dropout_rate, n_layers)
            for _, n_layers in enumerate(block_lengths)
        ]
        self._down_blocks = [
            blocks.DownBlock(filter_sizes[depth], dropout_rate)
            for depth, _ in enumerate(block_lengths)
        ]

    def __call__(self, inputs, is_training):
        stack = inputs
        skips = []

        for depth, _ in enumerate(self._block_lengths):
            stack, _ = self._dense_blocks[depth](stack, is_training)
            skips.append(stack)
            stack = self._down_blocks[depth](stack, is_training)

        return stack, skips


class TransitionUp:
    """Downsampling transition."""

    def __init__(self, block_lengths, filter_sizes, growth_rate, dropout_rate):

        self._block_lengths = block_lengths

        self._up_blocks = [
            blocks.UpBlock(filter_sizes[depth])
            for depth, _ in enumerate(block_lengths)
        ]
        self._concat_layers = [
            layers.Concatenate(axis=-1) for _ in block_lengths
        ]
        self._dense_blocks = [
            blocks.DenseBlock(growth_rate, dropout_rate, n_layers)
            for depth, n_layers in enumerate(block_lengths)
        ]

    def __call__(self, inputs, skips, is_training):
        blocked_stacks = inputs
        skips = skips[::-1]
        for depth, _ in enumerate(self._block_lengths):
            stack = self._up_blocks[depth](blocked_stacks)
            stack = self._concat_layers[depth]([stack, skips[depth]])
            stack, blocked_stacks = self._dense_blocks[depth](stack,
                                                              is_training)
        return stack


class FcDensenet:
    # pylint: disable=too-many-instance-attributes
    """FC-Densenet based model for segmentation.

        Reference written with Lasagne: https://github.com/SimJeg/FC-DenseNet

        Args:
            model_config: Dictionary, contains FC-Densenet config,
                - input_shape: list of integers, input shape.
                - block_lengths: List of integers,
                    number of convolutional layers in each dense block.
                - initial_n_filters: Integer,
                    number of filters in the first layer.
                - growth_rate: Integer, how fast number of filters grow.
                - middle_block_length: Integer, number of layers in
                    the middle dense block.
                - dropout_rate: Float in [0, 1], dropout rate.
                - last_layer: String, either "softmax" or "sigmoid".
                - num_classes: Howm many classes there are in prediction.
                : for defaults refer to the class __init__.
    """

    def __init__(self, config):
        # Number of filters in the first layer.
        initial_n_filters = config.get('initial_n_filters', 48)
        # How many filters to add at every layer in denseblock.
        growth_rate = config.get('growth_rate', 12)
        # How many layers in each dense block in down- and upsampling.
        block_lengths = config.get('block_lengths', [4, 5, 7, 10, 12])
        # How many layers in bottleneck dense block.
        middle_block_length = config.get('middle_block_length', 15)

        # Set filters size across th network.
        down_n_filters = [
            num_of_layers * growth_rate for num_of_layers in block_lengths
        ]
        down_n_filters = list(itertools.accumulate(down_n_filters))
        down_n_filters = [
            num_of_filters + initial_n_filters
            for num_of_filters in down_n_filters
        ]
        up_n_filters = [
            num_of_layers * growth_rate
            for num_of_layers in [middle_block_length] + block_lengths[:0:-1]
        ]

        # Set other params.
        dropout_rate = config.get('dropout_rate', 0.1)
        num_classes = config['num_classes']
        last_layer = config.get('last_layer', 'softmax')

        # Create structural parts.
        self._first_convo = layers.Conv2D(
            initial_n_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self._transition_down = TransitionDown(
            block_lengths,
            down_n_filters,
            growth_rate,
            dropout_rate,
        )
        self._bottleneck = blocks.DenseBlock(
            growth_rate,
            dropout_rate,
            middle_block_length,
        )
        self._transition_up = TransitionUp(
            block_lengths[::-1],
            up_n_filters,
            growth_rate,
            dropout_rate,
        )
        self._last_convo = layers.Conv2D(
            num_classes,
            kernel_size=(1, 1),
            padding='same',
            kernel_regularizer=regularizers.l2(1e-4))

        if last_layer == 'softmax':
            self._activation = layers.Softmax(axis=-1)
        elif last_layer == 'sigmoid':
            self._activation = layers.Activation(activation='sigmoid')
        else:
            raise ValueError('Last layer should be "sigmoid" or "softmax", '
                             f'given: {last_layer}.')

    def create_model(self, input_shape):
        """Return keras model of the FC-Densenet"""
        inputs = layers.Input(shape=input_shape)
        outputs = self._model(inputs)
        model = models.Model(inputs, outputs)
        return model

    def _model(self, input_tensor):
        """Returns densenet based segmentation model."""
        # First convo (no BN and ReLU in the source).
        stack = self._first_convo(input_tensor)
        # Downsampling.
        stack, skips = self._transition_down(stack, K.learning_phase())
        # Bottleneck.
        _, blocked_stacks = self._bottleneck(stack, K.learning_phase())
        # Upsampling.
        stack = self._transition_up(blocked_stacks, skips, K.learning_phase())
        # Last convo and activation.
        stack = self._last_convo(stack)
        output_tensor = self._activation(stack)
        return output_tensor
