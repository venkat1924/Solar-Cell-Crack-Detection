We provide cell images (`crack_segmentation`) and module images (`module_images`). There are 576 module images in the `module_images` folder. We cropped out cell images from those modules and sampled 1,837 images to train and evalute our model.

The details of the `crack_segmentation` folder are:

This dataset contains the training set, validation set and testing sets used train a semantic segmentation model for EL image analysis. The objects annotated are: 'crack', 'cross crack', 'busbar' and 'dark area'.

The tree structure of this dataset is:

```bash
crack_segmentation
├── train
│   ├── ann_json
│   │   ├── 005.json
│   │   ├── 007.json
│   │   └── ....json
|   ├── ann
│   │   ├── 005.png
│   │   ├── 007.png
│   │   └── ....png
│   └── img
│       ├── 005.jpg
│       ├── 007.jpg
│       └── ....jpg
├── val
│   ├── ann_json
│   │   ├── 001.json
│   │   ├── 002.json
│   │   └── ....json
|   ├── ann
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ....png
│   └── img
│       ├── 001.jpg
│       ├── 002.jpg
│       └── ....jpg
├── test_mix
│   ├── ann_json
│   │   ├── 023.json
│   │   ├── 034.json
│   │   └── ....json
|   ├── ann
│   │   ├── 023.png
│   │   ├── 034.png
│   │   └── ....png
│   └── img
│       ├── 023.jpg
│       ├── 034.jpg
│       └── ....jpg
└── test_crack
    ├── ann_json
    │   ├── 011.json
    │   ├── 012.json
    │   └── ....json
    |── ann
    │   ├── 011.png
    │   ├── 012.png
    │   └── ....png
    └── img
        ├── 011.jpg
        ├── 012.jpg
        └── ....jpg
```
`test_mix` contains cracked and intact cells. `test_crack` only contains cracked cells. These two sets have 133 images overlapped.

More details of loading data are at [PV-Vision](https://github.com/hackingmaterials/pv-vision). Please cite our work when using this public dataset.