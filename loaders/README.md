# Loaders

## Existing data loaders
* Severstal defect detection data loader
* Cityscapes segmentation data loader

## Add your own task
1. Write loader class inherited from `/loaders/data_loader.DataLoader` to load paths to images and labels.
2. Write a generator class inherited from `/generators/generator.Generator` (keras.utils.Sequence) to load images and labels from image paths.
3. Register your task in `/utils/general_utils.get_loader()` and make a config based on something from `/configs`.