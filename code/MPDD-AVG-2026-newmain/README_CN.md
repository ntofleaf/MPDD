## 🌐 Website

<div align="left">

[![Website](https://img.shields.io/badge/🌐%20Official%20Website-MPDD--AVG%202026-brightgreen?style=for-the-badge&logo=github&logoColor=white)](https://hacilab.github.io/MPDD-AVG-2026.github.io/index.html)

</div>

## 📄 Baseline Report

<div align="left">

[![Baseline Report](https://img.shields.io/badge/📑%20Baseline%20Report-MPDD--AVG%202026-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://github.com/hacilab/MPDD-AVG-2026/blob/main/MPDD_AVG%E2%80%94baseline.pdf)

</div>

## 🗂️ Dataset

<div align="left">

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-MPDD--AVG--2026-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/chasonfff/MPDD-AVG-2026/tree/main)

</div>

# MPDD-AVG Baseline

一个面向 `MPDD-AVG2026` 的多模态基线工程，支持：

- `Track1 = Elder`
- `Track2 = Young`
- 二分类任务：`label2`
- 三分类任务：`label3`
- 子赛道：`A-V+P`、`A-V-G+P`、`G+P`
- 编码器：`bilstm_mean`、`hybrid_attn`

当前开源版本默认提供的是二分类和三分类脚本。二分类和三分类训练时都会联合一个 `PHQ-9` 回归头，因此分类实验会同时输出分类指标和 `PHQ-9` 回归指标。

## 1. 项目特点

- 统一的 `train / val / test` 流程
- 官方 `test` 只在训练结束后评估一次
- 官方 `train` 内部再切 `val`
- 按 `ID` 划分，避免同一受试者跨 `train/val`
- 二分类和三分类默认联合训练 `分类头 + PHQ-9 回归头`
- 训练日志、历史曲线、checkpoint、结果汇总统一保存

分类任务输出 6 个指标：

- `Macro-F1`
- `ACC`
- `Kappa`
- `CCC`
- `RMSE`
- `MAE`

当前分类版本中：

- `binary` 使用 `label2`
- `ternary` 使用 `label3`
- `Macro-F1 / ACC / Kappa` 来自分类头
- `CCC / RMSE / MAE` 来自 `PHQ-9` 回归头

## 2. 目录结构

核心工程文件如下：

```text
MPDD_Avg/
├── config.json
├── dataset.py
├── metrics.py
├── train.py
├── test.py
├── train_val_split.py
├── models/
│   ├── __init__.py
│   ├── hybrid_temporal_encoder.py
│   └── torchcat_baseline.py
├── scripts/
│   ├── Track1/
│   │   ├── A-V-P/
│   │   ├── A-V-G+P/
│   │   └── G-P/
│   └── Track2/
│       ├── A-V-P/
│       ├── A-V-G+P/
│       └── G-P/
├── checkpoints/
├── logs/
└── MPDD-AVG2026/
```

说明：

- `scripts/Track1` 对应 `Elder`
- `scripts/Track2` 对应 `Young`
- `run_binary.sh` 对应二分类
- `run_ternary.sh` 对应三分类

## 3. 环境安装

建议环境：

- Linux
- Python `3.10` 或 `3.11`
- PyTorch 与本机 CUDA 对应版本

推荐使用 `conda`：

```bash
conda create -n mpddavg python=3.10 -y
conda activate mpddavg
pip install --upgrade pip
```

安装 PyTorch：

```bash
# 根据你的 CUDA 版本选择官方安装命令
pip install torch torchvision torchaudio
```

安装其余依赖：

```bash
pip install numpy scikit-learn
```

如果只在 CPU 上测试，可以运行时设置：

```bash
DEVICE=cpu
```

## 4. 数据集准备

本仓库默认数据路径是相对路径，要求数据放在仓库根目录下：

```text
MPDD-AVG2026/
├── MPDD-AVG2026-trainval/
│   ├── Elder/
│   │   ├── Audio/
│   │   ├── Video/
│   │   ├── IMU/
│   │   ├── split_labels_train.csv
│   │   └── descriptions_embeddings_with_ids.npy
│   └── Young/
│       ├── Audio/
│       ├── Video/
│       ├── IMU/
│       ├── split_labels_train.csv
│       └── descriptions_embeddings_with_ids.npy
└── MPDD-AVG2026-test/
    ├── Elder/
    │   ├── Audio/
    │   ├── Video/
    │   ├── IMU/
    │   └── split_labels_test.csv
    └── Young/
        ├── Audio/
        ├── Video/
        ├── IMU/
        └── split_labels_test.csv
```

标签文件至少应包含这些列：

- `ID`
- `split`
- `label2`
- `label3`
- `PHQ-9`

人格特征文件使用：

- `descriptions_embeddings_with_ids.npy`

## 5. 支持的模态与特征

子赛道定义：

- `A-V+P`：音频 + 视频 + 人格
- `A-V-G+P`：音频 + 视频 + 步态 + 人格
- `G+P`：步态 + 人格

支持的特征名：

- 音频：`mfcc`、`opensmile`、`wav2vec`
- 视频：`densenet`、`resnet`、`openface`

当前工程中，特征维度与数据集实际内容对应如下：

| 模态  | 特征                       | Elder / Track1 | Young / Track2 |
| ----- | -------------------------- | -------------: | -------------: |
| Audio | `mfcc` / `mfcc64`      |         `64` |         `64` |
| Audio | `opensmile`              |         `65` |         `65` |
| Audio | `wav2vec` / `wav2vec2` |        `768` |       `1024` |
| Video | `densenet`               |       `1000` |       `1000` |
| Video | `resnet`                 |       `1000` |       `1000` |
| Video | `openface`               |        `710` |        `710` |
| IMU   | `gait`                   |             12 |         `12` |

时序处理方式统一为：

- 所有时序模态 `audio / video / gait` 都会先读取原始 `[T, C]`
- 然后通过线性插值统一到固定长度 `target_t`
- 当前默认 `target_t = 128`
- `personality` 是定长向量，不做插值
- 这套处理对 `Track1 / Track2` 和三个子赛道都一致

也就是说：

- `A-V+P` 会插值 `audio + video`
- `A-V-G+P` 会插值 `audio + video + gait`
- `G+P` 只会插值 `gait`

当前脚本行为为：

- `scripts/Track1/A-V-P`
- `scripts/Track1/A-V-G+P`
- `scripts/Track2/A-V-P`
- `scripts/Track2/A-V-G+P`

以上四组脚本都会自动遍历 9 种 A/V 组合：

- 音频：`mfcc`、`opensmile`、`wav2vec`
- 视频：`densenet`、`resnet`、`openface`

而：

- `scripts/Track1/G-P`
- `scripts/Track2/G-P`

只跑 `G+P`，即 gait-only，不涉及 A/V 组合遍历。

数据加载器还兼容以下情况：

- `Audio / Video / IMU` 大小写目录
- `trainval / test` 自动配对
- Young 数据的事件命名差异

当前 `MPDD-AVG2026` 数据还需要注意：

- `Young test` 的 `Video/densenet`、`Video/resnet`、`Video/openface` 当前都是 `0` 字节空文件
- 因此 `Track2` 中所有含 `V` 的任务在 `test` 阶段，视频分支会退化为零输入
- `Track2` 的 `G+P` 不受这个问题影响
- `Young trainval` 的 `Video/openface` 也不是全量有效文件，使用时要结合日志中的有效样本数一起判断

## 6. 快速开始

### 6.1 直接使用现成脚本

示例：运行 `Track1 / Elder / A-V-G+P / 二分类`

```bash
bash scripts/Track1/A-V-G+P/run_binary.sh
```

这会顺序跑完 9 种 A/V 特征组合。

示例：运行 `Track1 / Elder / G+P / 三分类`

```bash
bash scripts/Track1/G-P/run_ternary.sh
```

示例：运行 `Track2 / Young / A-V+P / 二分类`

```bash
bash scripts/Track2/A-V-P/run_binary.sh
```

这也会顺序跑完 9 种 A/V 特征组合。

示例：运行 `Track2 / Young / A-V-G+P / 三分类`

```bash
bash scripts/Track2/A-V-G+P/run_ternary.sh
```

### 6.2 覆盖脚本默认超参数

脚本里常用超参数都可以通过环境变量覆盖，例如：

```bash
DEVICE=cpu EPOCHS=5 BATCH_SIZE=4 LR=1e-3 bash scripts/Track1/A-V-P/run_binary.sh
```

常见可覆盖参数：

- `DEVICE`
- `SEED`
- `EPOCHS`
- `BATCH_SIZE`
- `LR`
- `WEIGHT_DECAY`
- `HIDDEN_DIM`
- `DROPOUT`
- `PATIENCE`
- `MIN_DELTA`
- `TARGET_T`
- `PYTHON_BIN`

## 7. 手动命令行使用

### 7.1 训练

二分类示例：

```bash
python train.py \
  --track Track1 \
  --task binary \
  --subtrack A-V-G+P \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --device cuda
```

三分类示例：

```bash
python train.py \
  --track Track2 \
  --task ternary \
  --subtrack A-V+P \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --data_root MPDD-AVG2026/MPDD-AVG2026-trainval/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy \
  --device cuda
```

可选编码器：

- `bilstm_mean`
- `hybrid_attn`

### 7.2 测试指定 checkpoint（test_scripts/README.md）

**测试示例**

```bash
python test.py
   --checkpoint checkpoints/Track2/A-V-G+P/ternary/track2_ternary_A-V-G+P_bilstm_mean_wav2vec__resnet_log1p/best_model_*.pth
   --data_root MPDD-AVG2026/MPDD-AVG2026-test/Young
   --split_csv MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv
   --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy
```

### 7.3 只生成 train/val 划分预览

```bash
python train_val_split.py \
  --task ternary \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv \
  --save_path tmp/elder_ternary_split_preview.csv
```

## 8. 划分策略

当前划分规则如下：

- 只在官方 `train` 内部切 `val`
- 按 `ID` 划分，不让同一受试者同时出现在 `train` 和 `val`
- `binary` 按 `label2` 分层
- `ternary` 按 `label3` 分层
- 默认 `val_ratio = 0.1`
- （tips, 划分时可基于标签进行采样，保证各标签均能覆盖）

也就是说，只要数据和随机种子不变，`train/val` 划分就是固定可复现的。

## 9. 输出文件

训练输出会写入：

```text
checkpoints/{Track}/{SubtrackDir}/{task}/{experiment_name}/
logs/{Track}/{SubtrackDir}/{task}/{experiment_name}/
```

常见文件包括：

- `best_model_{timestamp}.pth`
- `result_{timestamp}.log`
- `history_{timestamp}.csv`
- `train_result_{timestamp}.json`
- `{experiment_name}.csv`

如果使用 `test.py` 单独评估，还会生成：

- `test_result_only_{timestamp}.json`

## 10. 指标与选模

分类任务验证和测试阶段会输出：

- `Macro-F1`
- `ACC`
- `Kappa`
- `CCC`
- `RMSE`
- `MAE`

best checkpoint 的选择依据为验证集上的：

- `binary / ternary`：`Macro-F1`
- `regression`：`CCC`

只有当验证集指标更好时，才会覆盖“当前最优模型”。

## 11. 复现实验建议

如果你想复现当前默认实验，建议优先直接运行 `scripts/` 里的脚本，因为这些脚本已经固定了：

- `track`
- `task`
- `subtrack`
- `audio_feature`
- `video_feature`
- 常用超参数

其中：

- `Track1` 使用 `config.json` 中的 Elder 默认路径
- `Track2` 脚本内部已经显式覆盖为 Young 路径
- 当前默认 `target_t = 128`

## 12. 常见问题

### 12.1 没有 GPU 可以运行吗？

可以，改成：

```bash
DEVICE=cpu bash scripts/Track1/A-V-P/run_binary.sh
```

### 12.2 如何切换特征？

直接改脚本，或者命令行传参：

```bash
python train.py --audio_feature mfcc --video_feature densenet ...
```

### 12.3 如何切换编码器？

```bash
python train.py --encoder_type hybrid_attn ...
```

### 12.4 为什么 `Track2` 的命令里数据路径更长？

因为 `config.json` 当前默认指向 `Track1 / Elder`。`Track2 / Young` 的脚本里已经显式传入 Young 的数据路径，所以直接运行脚本即可。

### 12.5 当前时序是怎么处理的？

当前不是硬截断前 `32` 帧，也不是随机裁剪。

- 所有时序模态都会把原始 `[T, C]` 线性插值到固定长度 `target_t`
- 当前默认 `target_t = 128`
- 如果需要，可以通过脚本环境变量 `TARGET_T` 或命令行 `--target_t` 覆盖

例如：

```bash
TARGET_T=256 bash scripts/Track1/G-P/run_binary.sh
```

### 12.6 为什么 `Track2` 含视频的测试结果要谨慎解释？

因为当前数据集中：

- `MPDD-AVG2026/MPDD-AVG2026-test/Young/Video/densenet`
- `MPDD-AVG2026/MPDD-AVG2026-test/Young/Video/resnet`
- `MPDD-AVG2026/MPDD-AVG2026-test/Young/Video/openface`

下的测试文件目前都是空文件。

因此：

- `Track2 / A-V+P`
- `Track2 / A-V-G+P`

在 `test` 阶段的视频分支没有真实视频信息，当前工程会自动补零以保证流程跑通。
