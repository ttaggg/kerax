""""Generators for Severstal defect detection dataset."""

import numpy as np
from keras import preprocessing

from generators import generator
from utils import image_utils


def _load_image(path):
    return np.asarray(
        preprocessing.image.load_img(path, color_mode='grayscale')) / 255.


class SegmentationGenerator(generator.Generator):
    """Custom generator for segmentation.

    Args:
        :same as parent class.
        kwargs: all generator-specific parameters:
            include_background: Bool, whether add background to masks
                as separate channel (to support sigmoid activation).
                If you do not include background, do not forget to
                change 'last_layer' and 'num_classes' in the model
                config accordingly.
    """

    def __init__(self, dataset, batch_size, augmentations, is_training,
                 **kwargs):
        self._include_background = kwargs['include_background']
        super().__init__(dataset, batch_size, augmentations, is_training)

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch = [self._to_images(el) for el in batch]
        x_data, y_data = tuple(zip(*batch))
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        return x_data, y_data

    def _to_images(self, pair):
        image_path, mask_list = pair
        image = _load_image(image_path)
        masks = [
            image_utils.rle_to_mask(el[1], image.shape) for el in mask_list
        ]
        image, masks = self._augment((image, masks))
        image = np.expand_dims(image, axis=-1)
        mask = np.stack(masks, axis=-1)

        if self._include_background:
            defect_probs = np.sum(mask, axis=-1, keepdims=True)
            clean_probs = np.ones_like(defect_probs) - defect_probs
            mask = np.concatenate([clean_probs, mask], axis=-1)

        return image, mask

    def _augment(self, pairs):
        image, masks = pairs
        data = {'image': image, 'masks': masks}
        aug_data = self._augmentations(**data)
        return aug_data['image'], aug_data['masks']


class ClassificationGenerator(generator.Generator):
    """Custom generator for classification.

    Args:
        :same as parent class.
        kwargs: all generator-specific parameters.
            num_channels: Num channels in the image.
    """

    def __init__(self, dataset, batch_size, augmentations, is_training,
                 **kwargs):
        self._num_channels = kwargs['num_channels']
        super().__init__(dataset, batch_size, augmentations, is_training)

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch = [self._to_data(el) for el in batch]
        x_data, y_data = tuple(zip(*batch))
        return np.asarray(x_data), np.asarray(y_data)

    def _to_data(self, pair):
        image_path, mask_list = pair
        image = _load_image(image_path)

        masks = [
            image_utils.rle_to_mask(elem[1], image.shape) for elem in mask_list
        ]
        # Need to process masks too, because defects
        # can be cut out during augmentation.
        image, masks = self._augment((image, masks))
        image = np.expand_dims(image, axis=-1)
        if self._num_channels == 3 and image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        defect_types = []
        for i, mask in enumerate(masks):
            if np.sum(mask) != 0:
                defect_types.append(i)

        labels = np.zeros(len(masks))
        labels[defect_types] = 1.

        return image, labels
