
# Explainable video depression detection

## Reproduce the experimental results


Explanation: Directly reproduce the original experimental results,

It is necessary to ensure the same torch version and GPU model, because the same random seed may produce different results in each forward pass under different torch versions or GPU models, making it difficult to directly reproduce the original experimental results.
The same dataset processing method should be used, such as extracting 224x224-sized face images using Openface2, and converting BMP-format images to JPG-format images.


- [Download Northwind dataset](https://drive.google.com/file/d/1PKQW2hZtMdsRo29FHqMKeU1jgxdd7ak_/view?usp=sharing)
- [Download Freeform dataset](https://drive.google.com/file/d/1r2Hj5TPZIKGAIvBIEylPu9cBqAE5Bhl6/view?usp=sharing)
- [Download Avce2013 dataset](https://drive.google.com/file/d/1sj6ntNcGwEBNk73elFn7pm8EUvcQbCy6/view?usp=sharing)

### Experiments Enveriment：

- Northwind dataset： torch '2.0.0+cu118'; GPU 3090
- Freeform dataset： torch '2.0.0+cu118; GPU 3090
- Avec2013 dataset： torch '2.0.0+cu118'; GPU 4090


## Reproduce 

Take Northwind dataset as an example

### Install env

On a machine with a 3090 GPU, install torch '2.0.0+cu118' and the code repository, as follows:

#### Conda
```shell
# create and activate env
conda create -n exp
conda activate exp
# install torch 2.0 with CUDA 11.8 from pytorch official source
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# installation tools for MM lib
pip install openmim
git clone https://github.com/liaorongfan/Explainable-Depression-Detection
cd mmpretrain
mim install -e .
```

#### Venv
```shell
# create env
virtualenv -n exp
# activate the env
. /.venv/source/bin/activate
# install torch 2.0 with CUDA 11.8 from pytorch official source
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# installation tools for MM lib
pip install openmim
git clone https://github.com/liaorongfan/Explainable-Depression-Detection
cd mmpretrain
mim install -e .

```

Detailed Reference： [mmpretrain installation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) 

### Download dataset

Take Northwind dataset as an example, down the dataset into `data` directory，
```
data/avec_2014_northwind
        ├── Development
        ├── Testing
        ├── Training
        ├── extracted_features
        └── label_dict.jsondata
```

### Training steps

#### Segment level model training
```shell
python tools/train.py configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.84.py
```

After training, the model should achieve `rmse=7.84`. In the original experiment, to further reduce the RMSE, an additional step of fine-tuning was performed, as follows:

```shell
python tools/train.py \
configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.72_finetune_from_7.84.py \
--resume /path/to/trained_model_with_rmse7.84
```
After fine-tuning, the model should achieve `rmse=7.72`.

#### Video level model training

First, extract the features of the segment.
```shell
python tools/extract.py \
configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.84.py \
path/to/trained_model_with_rmse7.72
```

Then, use the following configuration file to train the video-level model.
```shell
python tools/train.py configs/exp_reproduce/eat_video_level_northwind_7.17.py
```
After training, the model should achieve rmse=7.17

## Notation

If the random seed is not changed for each training session, the experimental results should be the same.

If the experiment cannot be reproduced exactly in one go, set the random seed in the configuration file to null (as shown below), and train multiple times in the current environment to find a suitable initialization seed.

```python
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=True)
```
When the seed is set to None, a random seed will be generated for each training session and recorded in the log file, as shown below:
```
Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 899834960
    deterministic: True
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
```

To reproduce the best training results in the current environment, you need to fix the random seed in the configuration file according to the log file, as shown below:
```python
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=899834960, deterministic=True)
```
