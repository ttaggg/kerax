# Models

## Existing models

* Custom models for segmentation
    * FC-Densenet
    * UNet
* `keras.applications` models for classification and regression
    * ResNet-18, ResNet-34, ResNet-50
    * Densenet-121


## Add your own model

* Add custom model or any model from `keras.applications` to `model_hub.py` and modify `model_hub.create_model()`.