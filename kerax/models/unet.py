"""UNet.

Original paper: Ronneberger et al. 2015.
"""
import keras.backend as K
from keras import layers
from keras import models
from keras import regularizers

from kerax.models import unet_blocks as blocks


class TransitionDown:
    """Downsampling transition."""

    def __init__(self, num_layers, filter_sizes, dropout_rate):
        self._down_blocks = [
            blocks.DownBlock(filter_sizes[depth], dropout_rate)
            for depth in range(num_layers)
        ]

    def __call__(self, inputs, is_training):
        outputs = inputs
        skips = []
        for down_block in self._down_blocks:
            outputs, output_nopool = down_block(outputs, is_training)
            skips.append(output_nopool)
        return outputs, skips


class TransitionUp:
    """Downsampling transition."""

    def __init__(self, num_layers, filter_sizes, dropout_rate):
        self._up_blocks = [
            blocks.UpBlock(filter_sizes[depth], dropout_rate)
            for depth in range(num_layers)
        ]

    def __call__(self, inputs, skips, is_training):
        skips = skips[::-1]
        outputs = inputs
        for i, up_block in enumerate(self._up_blocks):
            outputs = up_block(outputs, skips[i], is_training)
        return outputs


class Unet:
    # pylint: disable=too-many-instance-attributes
    """UNet based model for segmentation.

        Args:
            model_config: Dictionary, contains FC-Densenet config,
                - initial_n_filters: Integer,
                    number of filters in the first layer.
                - middle_n_filters: Integer, number of filters in
                    the middle layer.
                - num_layers: Integer,number of layers.
                - use_batchnorm: Bool, whether use batch normalization.
                - down_dropout_rate: Float in [0, 1], dropout rate
                    for downsampling.
                - up_dropout_rate: Float in [0, 1], dropout rate
                    for upsampling.
                - last_layer: String, either "softmax" or "sigmoid".
                - num_classes: Howm many classes there are in prediction.
                : for defaults refer to the class __init__.
    """

    def __init__(self, config):

        num_layers = config.get('num_layers', 4)
        down_n_filters = [
            config.get('initial_n_filters', 64) * (2**i)
            for i in range(num_layers)
        ]
        up_n_filters = down_n_filters[::-1]

        down_dropout_rate = config.get('down_dropout_rate', 0.2)
        up_dropout_rate = config.get('up_dropout_rate', 0.0)
        num_classes = config['num_classes']
        last_layer = config.get('last_layer', 'softmax')

        # Create structural parts.
        self._transition_down = TransitionDown(
            num_layers,
            down_n_filters,
            down_dropout_rate,
        )
        self._bottleneck = blocks.ConvoBlock(down_n_filters[-1] * 2)
        self._transition_up = TransitionUp(
            num_layers,
            up_n_filters,
            up_dropout_rate,
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
        """Return keras model of UNet."""
        input_layer = layers.Input(shape=input_shape)
        output_layer = self._model(input_layer)
        model = models.Model(input_layer, output_layer)
        return model

    def _model(self, input_tensor):
        """Creates UNet architecture."""
        stack = input_tensor
        # Downsampling.
        stack, skips = self._transition_down(stack, K.learning_phase())
        # Bottleneck.
        stack = self._bottleneck(stack, K.learning_phase())
        # Upsampling.
        stack = self._transition_up(stack, skips, K.learning_phase())
        # Last convo and activation.
        stack = self._last_convo(stack)
        output_tensor = self._activation(stack)
        return output_tensor
