
## How to run

#### Training
```bash
python -m kerax.train \
    --output_dir <path/to/output_dir> \
    --config <path/to/config.py>
```

If you use bazel, then (tested with `bazel 3.4.1`).
```bash
bazel run //:train -- \
    --output_dir <path/to/output_dir> \
    --config <path/to/config.py>
```

#### Prediction
```bash
python -m kerax.predict \
    --output_dir <path/to/output_dir> \
    --config <path/to/config.py> \
    --load_saved_model <path/to/checkpoint.hdf5>
```

```bash
bazel run //:predict -- \
    --output_dir <path/to/output_dir> \
    --config <path/to/config.py> \
    --load_saved_model <path/to/checkpoint.hdf5>
```

## Supported tasks

* Cityscapes segmentation for [Cityscapes dataset](cityscapes-dataset.com).
* Sevestal segmentation for [Severstal defect detection challenge](https://www.kaggle.com/c/severstal-steel-defect-detection). 
* Sevestal classification for [Severstal defect detection challenge](https://www.kaggle.com/c/severstal-steel-defect-detection). 
* MAuto regression task for speed estimation.
* Easy to add your own task by adding your custom loader and generator, see README in `/loaders` and `/generators`. It took me 1 hour to add Cityscapes to pipeline and 24 hours to download their dataset.


If you are going to change already written utils or loaders / generators, please run to check if tests are working.
```bash
bazel test --cache_test_results=no //...
```

## Config

Examples of configs are given in `/configs` directory.
Also minimal and maximal (all posible features) configs are providied in `/testdata` directory.

## Project structure

Without bazel-related files, READMEs and unit-test related data. 
```
├── flags.py
├── generators
│   ├── cityscapes_generator.py
│   ├── generator.py
│   ├── mauto_generator.py
│   ├── severstal_generator.py
│   └── tests
│       └── <...>
├── loaders
│   ├── cityscapes_data_loader.py
│   ├── data_loader.py
│   ├── mauto_data_loader.py
│   ├── severstal_data_loader.py
│   └── tests
│       └── <...>
├── models
│   ├── fc_densenet.py
│   ├── fcd_blocks.py
│   ├── model_hub.py
│   ├── unet.py
│   └── unet_blocks.py
├── predict.py
├── testdata
│   └── <...>
├── train.py
└── utils
    ├── callbacks.py
    ├── general_utils.py
    ├── image_utils.py
    ├── losses.py
    ├── metrics.py
    └── tests
        └── <...>
```