
# Explainable video depression detection

## Data preparation
### From raw AVEC 2013 and 2014 dataset

We utilize OpenFace 2.0 tools to extract face images from raw videos with the following steps:

1. Face Image Extraction: The resolution for face images is set to 224x224 during extraction.

2. Image Format Conversion: To save storage space, the extracted face images are converted from BMP to JPG format using OpenCV APIs.
The script for this conversion can be found [here](./tools/convert_bmp_to_jpg.py).

3. Frame Selection for Training: For the AVEC 2013 dataset, only the first 10,000 frames from each video are used for training. This decision ensures quicker training, as some videos exceed 20 minutes in duration.



### Processed data
For the convinience of experiment reproduce, processed data can be found [here](./path/to/google/drive)


## Model Training


The training process comprises two stages:

**Stage 1: Spatial-Temporal Feature Extraction**

In this phase, a spatial-temporal feature extractor is trained to analyze and represent video segments effectively.

**Stage 2: Weakly Supervised Learning**

Weakly supervised learning is employed to enhance the discrimination of the video segments, further improving their depression related representation.


### On AVEC 2014 dataset
#### Northwind dataset
First training a feature extractor by
```shell
python tools/train.py configs/first_stage/eat_seq32_avec_northwind_ordinal_cls.py
```
After training the extractor, it can be used for feature extaction by:
```shell
python tools/extract.py configs/first_stage/eat_seq32_avec_northwind_ordinal_cls.py /path/to/the_best_rmse_model.pth 
```
This will generate feature dataset in following file structure by default.
```
./data/avec
        └── extracted_features
            ├── Development
            │   ├── 205_1_Development.pkl
            │   ├── 206_1_Development.pkl

            ├── Testing
            │   ├── 203_2_Testing.pkl
            │   ├── 206_2_Testing.pkl

            └── Training
                ├── 203_1_Training.pkl
                ├── 205_2_Training.pkl
```
At last, the second stage can be trained by using:

```shell
python tools/train.py configs/second_stage/vat_anchor.py
```

### Freeform and AVEC_2013 dataset
The same as training on northwind dataset, the training process is as follow:
```shell
python tools/train.py configs/first_stage/eat_seq32_avec_{}_ordinal_cls.py
```
```shell
python tools/extract.py configs/first_stage/eat_seq32_avec_{freeform,avec_2013}_ordinal_cls.py 
```
```shell
python tools/train.py configs/second_stage/vat_anchor.py
```

## Performance


|   Dataset  |   BaseLine   |    Stage 1   |   Stage 2   |
|:----------:|:-----------:|:------------:|:-----------:|
| Northwind  | 8.35 | 7.74 | 7.17 |
| Freefrom   | -- | -- | -- |
| AVEC 2013  | -- | -- | -- |
