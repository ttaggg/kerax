"""Minimal config that contains all required fields.

If the data has different dimension from input_shape,
train and test augmentation with crops also need to be specified.
"""
# pylint: disable=invalid-name
# yapf: disable

model = {
    'model_name': 'fc_densenet',
    'input_shape': [128, 800, 1],
    'num_classes': 5,
}

data = {
    'data_path': 'testdata/test_set.csv',
    'task': 'severstal_segmentation',
}

training = {
    'batch_size': 4,
    'num_epochs': 5,
    'loss_function': 'binary_cross_entropy',
}
