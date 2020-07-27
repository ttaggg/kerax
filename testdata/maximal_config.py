"""Maximal config for segmentation with all possible parameteres."""
# pylint: disable=invalid-name
# yapf: disable

# Other models supported are: 'unet', 'resnet50', 'densenet121'.
# Any model can be added to pipeline in models/model_hub.py.
# Parameters depend on the model choice.
model = {
    'model_name': 'fc_densenet',
    'input_shape': [256, 512, 3],
    'initial_n_filters': 32,
    'block_lengths': [4, 4, 4, 6, 8],
    'middle_block_length': 8,
    'growth_rate': 12,
    'dropout_rate': 0.1,
    'last_layer': 'softmax',
    'num_classes': 35,
}

data = {
    'data_path':
        '<path/to/dir/with/gtFine_trainvaltest_and_leftImg8bit_trainvaltest>',
    # Currently supported tasks are:
    # cityscapes_segmentation, severstal_segmentation, severstal_classification
    'task': 'cityscapes_segmentation',
    'n_folds': 20,
    'fold': 0,
    # All augmentations are from 'albumentations' library.
    # Currently augmentations are limited to the following list,
    # but anything from albumentations library can be added to
    # generators/generator.py.
    # Any arguments accepted by particular augmentation can be used.
    'augmentation': {
        'train': {
            'Resize': {
                'height': 512,
                'width': 1024,
                'always_apply': True,
            },
            'RandomCrop': {
                'height': 256,
                'width': 512,
                'always_apply': True,
            },
            'HorizontalFlip': {'p': 0.5},
            'VerticalFlip': {'p': 0.5},
            'RandomBrightness': {
                'p': 0.5,
                'limit': 0.2
            },
            'ShiftScaleRotate': {
                'p': 0.5,
                'shift_limit': 0.1625,
                'scale_limit': 0.6,
                'rotate_limit': 0
            },
        },
        'test': {
            'RandomCrop': {
                'height': 128,
                'width': 800,
                'always_apply': True,
            },
        }
    }
}

training = {
    'batch_size': 4,
    'num_epochs': 5,
    # Also another metric is 'mcc': Matthews correlation coefficient
    # available for classification task.
    'metrics': ['dice'],
    'loss_function': 'binary_cross_entropy',
    'optimizer': {
        # Also 'rmsprop', anything can be added in general_utils.get_optimizer()
        'optimizer_name': 'adam',
        'learning_rate': 1e-3,
        # Any parameter that can be used as an input to keras.optimizer.*
        'params': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': None,
            'decay': 0.,
            'amsgrad': False
        },
    },
    'callbacks': {
        # One more callback is used for classification task:
        # MultiClassifierEpochCallback returns f1_score, precision, recall.
        'custom_callbacks': {
            # Logs images to tensorboard.
            'ImageLogger': {
                'batches_to_log': 20,
                'update_freq': 1000,
            },
            # Calculates Dice score on the whole test set.
            'SegmentationDiceEpochCallback': {
                'batches_to_log': 20,
                'update_freq': 1000,
                'cut_background_index': 'first', # can be 'last'
            }
        },
        # Parameters for ReduceLROnPlateau and ReduceLROnPlateau callbacks.
        'monitor': 'val_dice_epoch',
        'monitor_mode': 'max',
        'patience': 2,
        # how often to update Tensorboard:
        # should divide 'update_freq' in callbacks above.
        'tb_update_freq': 500,
    }
}
