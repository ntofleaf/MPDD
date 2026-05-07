
# Explainable video depression detection

## 依照配置文件复现实验结果

说明：直接复现原来的实验结果，
- 需要保证相同的 `torch 版本`， `GPU型号`，因为相同的`随机种子`在不同`torch版本`或`GPU型号`下，每次前向运行的结果会有不同，导致难以直接复现原来的实验结果。
- 相同的数据集处理方法，Openface2 提取224*224大小的人脸图片,并将bmp格式的图片转存为jpg格式的图片
- [Northwind数据集下载](https://drive.google.com/file/d/1PKQW2hZtMdsRo29FHqMKeU1jgxdd7ak_/view?usp=sharing)
- [Freeform数据集下载](https://drive.google.com/file/d/1r2Hj5TPZIKGAIvBIEylPu9cBqAE5Bhl6/view?usp=sharing)
- [Avce2013数据集下载](https://drive.google.com/file/d/1sj6ntNcGwEBNk73elFn7pm8EUvcQbCy6/view?usp=sharing)

### 实验环境：

- Northwind数据集： torch '2.0.0+cu118'; GPU 3090
- Freeform 数据集： torch '1.13.0+cu117; GPU 3090
- Avec2013 数据集： torch '2.0.0+cu118'; GPU 4090

(注：当前正在统一三个数据集的实验环境)

## 复现实验

以Northwind数据集为例

### 环境安装
在配置3090显卡的机器上，安装torch '2.0.0+cu118'和代码库，如下

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

详细参考： [mmpretrain 安装](https://mmpretrain.readthedocs.io/en/latest/get_started.html) 

### 数据集下载

以Northwind数据集为例, 下载数据集到`data`目录下，
```
data/avec_2014_northwind
        ├── Development
        ├── Testing
        ├── Training
        ├── extracted_features
        └── label_dict.jsondata
```

### 训练步骤

#### Segment level 模型训练
```shell
python tools/train.py configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.84.py
```
训练结束后，应该会得到 `rmse=7.84` 的模型，在原来的实验中为进一步降低rmse，增加了一步 finetune， 方式如下：
```shell
python tools/train.py \
configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.72_finetune_from_7.84.py \
--resume /path/to/trained_model_with_rmse7.84
```

`Finetune`结束后应该会得到 `rmse=7.72` 的模型。
#### Video level 模型训练

首先提取，segment 的特征

```shell
python tools/extract.py \
configs/exp_reproduce/eat_seq32_avec_northwind_ordinal_cls_7.84.py \
path/to/trained_model_with_rmse7.72
```

然后使用如下配置文件训练video level的模型。

```shell
python tools/train.py configs/exp_reproduce/eat_video_level_northwind_7.17.py
```
训练结束后应该会得到 rmse=7.17的模型。

## 注意事项
如果每次训练不修改`随机种子`的话，所得到的实验结果应该是相同的。

如果实验不能一次复现原结果，需要将配置文件中的随机种子设置为空（如下），在当前环境中多训练几次，来找到较合适的初始化种子
```python
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=True)
```
当`seed`设置为`None`时，每次训练会随机生成一个随机种子,并记录在日志文件中，如下所示：
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
复现当前环境中较好的训练结果，需要使用根据`日志文件`，固定`配置文件`中的随机种子，如下：

```python
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=899834960, deterministic=True)
```
