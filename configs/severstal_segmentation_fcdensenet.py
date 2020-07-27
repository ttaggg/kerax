"""Fully-convolutional Densenet: medium size.

Defect detection training on the Severstal data.
"""
# pylint: disable=invalid-name
# yapf: disable

model = {
    'model_name': 'fc_densenet',
    'input_shape': [128, 800, 1],
    'block_lengths': [4, 4, 4, 6, 8],
    'growth_rate': 12,
    'dropout_rate': 0.1,
    'last_layer': 'softmax',
    'num_classes': 5,
}

training = {
    'batch_size': 4,
    'num_epochs': 20,
    'metrics': ['dice'],
    'loss_function': 'binary_cross_entropy',
    'optimizer': {
        'optimizer_name': 'adam',
        'learning_rate': 5e-4,
    },
    'callbacks': {
        'custom_callbacks': {
            'ImageLogger': {
                'update_freq': 1000,
                'batches_to_log': 20,
            },
            'SegmentationDiceEpochCallback': {
                'update_freq': 1000,
                'cut_background_index': 'first',
            }
        },
        'monitor': 'val_dice_epoch',
        'monitor_mode': 'max',
    }
}

data = {
    'data_path': '<path/to/dir/with/csv_and_images>',
    'task': 'severstal_segmentation',
    'n_folds': 20,
    'fold': 0,
    'augmentation': {
        'train': {
            'RandomCrop': {
                'height': 128,
                'width': 800,
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
