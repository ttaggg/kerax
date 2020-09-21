""""Generators for Cityscapes segmentation dataset."""

import numpy as np
from keras import preprocessing

from kerax.generators import generator


def _load_image(path):
    return np.asarray(preprocessing.image.load_img(path), dtype=np.uint8)


def _load_mask(path):
    return np.array(
        preprocessing.image.load_img(path, color_mode='grayscale'),
        dtype=np.uint8)


class CityscapesGenerator(generator.Generator):
    """Custom generator for segmentation.

    Args:
        :same as parent class.
    """

    def __init__(self, dataset, batch_size, augmentations, is_training,
                 **kwargs):
        super().__init__(dataset, batch_size, augmentations, is_training)
        # Cityscapes dataset has 33 classes + 0 for 'unlabeled'.
        # We set all ignored classes to 0.
        ignore_labels = kwargs['ignore_labels']
        self._num_classes = 34 - len(ignore_labels)

        self._update_labels_dict = {0: 0}
        minimal_inx = 1
        for i in range(1, 34):
            if i in ignore_labels:
                self._update_labels_dict[i] = 0
            else:
                self._update_labels_dict[i] = minimal_inx
                minimal_inx += 1
        # Of course it is, just sanity check.
        assert self._num_classes == len(set(self._update_labels_dict.values()))

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch = [self._to_images(el) for el in batch]
        x_data, y_data = tuple(zip(*batch))
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        return x_data, y_data

    def _to_images(self, pair):
        image_path, mask_path = pair
        image = _load_image(image_path)
        mask = _load_mask(mask_path)

        for key in self._update_labels_dict:
            mask[mask == key] = self._update_labels_dict[key]

        mask = np.eye(self._num_classes)[mask]

        image, mask = self._augment((image, mask))
        return image, mask

    def _augment(self, pairs):
        image, mask = pairs
        data = {'image': image, 'mask': mask}
        aug_data = self._augmentations(**data)
        return aug_data['image'], aug_data['mask']

    @property
    def colormap(self):
        """Colormap for labels, only used for segmentation."""
        palette = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                   (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232),
                   (250, 170, 160), (230, 150, 140), (70, 70, 70),
                   (102, 102, 156), (190, 153, 153), (180, 165, 180),
                   (150, 100, 100), (150, 120, 90), (153, 153, 153),
                   (153, 153, 153), (250, 170, 30), (220, 220, 0),
                   (107, 142, 35), (152, 251, 152), (70, 130, 180),
                   (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                   (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100),
                   (0, 0, 230), (119, 11, 32), (0, 0, 142)]

        num_colors = self[0][1].shape[-1]
        colormap = np.zeros((num_colors, 3), dtype=int)
        for i in range(num_colors):
            colormap[i, ...] = palette[self._update_labels_dict[i]]
        return colormap
