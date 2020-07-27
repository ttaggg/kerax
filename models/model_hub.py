"""List of available models."""
# TODO(ttaggg): refactor this file.
# TODO(ttaggg): cover models with tests.

from absl import logging
from keras import models
from keras import layers
from keras import applications

from models import fc_densenet
from models import unet


def create_model(model_config):
    """Choose the model."""

    model_name = model_config['model_name']
    models_dict = {
        'densenet121': densenet121_fn,
        'fc_densenet': fcdensenet_fn,
        'resnet50': resnet50_fn,
        'unet': unet_fn,
    }
    if model_name not in models_dict:
        raise ValueError(f'Unknown model name: {model_name}'
                         f'Available names are: {list(models_dict.keys())}')

    logging.info('Model is created.')
    return models_dict[model_name](model_config)


def fcdensenet_fn(model_config):
    """Returns FC-Densenet model.

    Args:
        model_config: Dictionary, contains FC-Densenet config,
            for description refer to the class header.
    Returns:
        model: Keras model.
    """
    model = fc_densenet.FcDensenet(model_config).create_model(
        model_config['input_shape'])
    return model


def densenet121_fn(model_config):
    """Densenet-121, untested."""

    last_layer = model_config.get('last_layer', 'softmax')
    weights = model_config.get('weights', None)
    base_model = applications.densenet.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=model_config['input_shape'],
        pooling='avg')

    net = base_model.output
    net = layers.Flatten()(net)
    output = layers.Dense(model_config['num_classes'])(net)
    if last_layer == 'softmax':
        output = layers.Softmax(axis=-1)(output)
    elif last_layer == 'sigmoid':
        output = layers.Activation(activation='sigmoid')(output)

    model = models.Model(base_model.input, output)
    return model


def resnet50_fn(model_config):
    """ResNet50, untested."""

    last_layer = model_config.get('last_layer', 'softmax')
    weights = model_config.get('weights', None)
    base_model = applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=model_config['input_shape'],
        pooling='avg')

    net = base_model.output
    output = layers.Dense(model_config['num_classes'])(net)
    if last_layer == 'softmax':
        output = layers.Softmax(axis=-1)(output)
    elif last_layer == 'sigmoid':
        output = layers.Activation(activation='sigmoid')(output)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model


def unet_fn(model_config):
    """UNet.

    Args:
        model_config: Dictionary, contains UNet config,
            for description refer to the class header.
    Returns:
        model: Keras model.
    """
    model = unet.Unet(model_config).create_model(model_config['input_shape'])
    return model
