"""Classification task on the Severstal data."""
# pylint: disable=invalid-name
# yapf: disable

model = {
    'model_name': 'resnet50',
    'weights': 'imagenet',
    'input_shape': [128, 800, 3],
    'num_classes': 5,
}

training = {
    'batch_size': 16,
    'num_epochs': 20,
    'loss_function': 'cross_entropy',
    'optimizer': {
        'optimizer_name': 'adam',
        'learning_rate': 5e-4,
    },
    'callbacks': {
        'custom_callbacks': {
            'MultiClassifierEpochCallback': {
                'update_freq': 1000,
            }
        },
        'monitor': 'val_f1_score',
        'monitor_mode': 'max',
    }
}

data = {
    'data_path': '<path/to/dir/with/csv_and_images>',
    'labels_path': '<path/to/dir/with/csv_and_images>',
    'task': 'severstal_classification',
    'n_folds': 20,
    'fold': 0,
    'num_channels': 3,
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
