"""UNet.

Segmentation for Cityscapes data.
"""
# pylint: disable=invalid-name
# yapf: disable

model = {
    'model_name': 'unet',
    'input_shape': [256, 512, 3],
    'dropout_rate': 0.1,
    'last_layer': 'softmax',
    'num_classes': 35,
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
            }
        },
        'monitor': 'val_dice_epoch',
        'monitor_mode': 'max',
    }
}

data = {
    'data_path':
        '<path/to/dir/with/gtFine_trainvaltest_and_leftImg8bit_trainvaltest>',
    'task': 'cityscapes_segmentation',
    'n_folds': 20,
    'fold': 0,
    'augmentation': {
        'train': {
            'Resize': {
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
            'Resize': {
                'height': 256,
                'width': 512,
                'always_apply': True,
            },
        }
    }
}
