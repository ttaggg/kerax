# Utils

## `general_utils.py`

* initialize directory for training
* check config files for missing fields
* return appropriate loader, optimizer, loss function, metrics, callbacks list, load model

## `losses.py`
* list of custom loss functions:
* Cross-entropy and cross-entropy with Dice
* Binary cross-entropy and binary cross-entropy with Dice

## `metrics.py`
* Dice metric for image segmentation
* Matthews correlation coefficient (MCC)

## `callbacks.py`
* Segmentation:
    * ImageLogger: log image, ground truth and predicted mask to tensorboard.
    * SegmentationDiceEpochCallback: calculate Dice score on the whole test dataset, update tensorboard.
* Classification:
    * MultiClassifierEpochCallback: calculate f1-score, precision and recall on the whole test dataset, update tensorboard
* Others:
    * CustomTensorBoardCallback: override keras.callbacks.TensorBoard to make it work with other callbacks.

## `image_utils.py`
* generate old tensorflow image summary
* convert mask to RLE.
* convert RLE to mask.
* Colorify predicted masks for visualization.
