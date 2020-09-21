# pylint: disable=invalid-name
# yapf: disable

model = {
    'model_name': 'resnet50',
    'input_shape': [384, 512, 3],
    'last_layer': 'leaky_relu',
    'num_classes': 1,
}

training = {
    'batch_size': 16,
    'num_epochs': 30,
    'metrics': ['mse', 'mae'],
    'loss_function': 'huber_loss',
    'optimizer': {
        'optimizer_name': 'adam',
        'learning_rate': 2e-4,
    },
    'callbacks': {
        'custom_callbacks': {
            'RegressionEpochCallback': {
                'update_freq': 1000,
                'rolling_mean': 41,
            }
        },
        'monitor': 'val_smooth_mse_epoch',
        'monitor_mode': 'min',
        'lr_factor': 0.75
    }
}

data = {
    'data_path':
        '<path/to/frames>',
    'labels_path':
        '<path/to/labels.txt>',
    'task': 'mauto_regression',
    'n_folds': 10,
    'fold': 0,
    'random_state': None,
    'shuffle': False,
    'augmentation': {
        'train': {
            'Resize': {
                'height': 384,
                'width': 512,
                'always_apply': True,
            },
            'Normalize': {
                'always_apply': True,
                'mean': (0.02942127, 0.0382734,  0.01791393),
                'std': (0.07160185, 0.08250787, 0.04437295),
            }
        },
        'test': {
            'Resize': {
                'height': 384,
                'width': 512,
                'always_apply': True,
            },
            'Normalize': {
                'always_apply': True,
                'mean': (0.02942127, 0.0382734,  0.01791393),
                'std': (0.07160185, 0.08250787, 0.04437295),
            }
        }
    }
}