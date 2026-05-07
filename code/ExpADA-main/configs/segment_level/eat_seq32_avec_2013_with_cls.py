_base_ = [
    '../_base_/models/eat.py',
    '../_base_/datasets/avec.py',
    '../_base_/schedules/avec_adam.py',
    '../_base_/default_runtime.py',
]

INPUT_SEQUENCE_LEN, STEPS = 32, 16
BATCH_SIZE, WORKERS = 16, 8

model = dict(
    # pretrained="work_dirs/seq32_cls/last_checkpoint.pth",
    neck=dict(
        type="VatNeck",
        seq_len=INPUT_SEQUENCE_LEN,
        head=8,
        spatial_h=7,
        spatial_w=7,
    ),
    head=dict(
        type="LinearClsRegHead",
        in_channels=1024,
        num_classes=4,
        loss=dict(type="RegressionLoss", loss_weight=1.0),
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs', algorithm_keys=["subj_id", 'gt_cls_label',]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs', algorithm_keys=["subj_id", 'gt_cls_label',]),
]


# data loader and dataset settings
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    dataset=dict(
        data_root='./data/avec_2013',
        seg_len=INPUT_SEQUENCE_LEN,
        steps=STEPS,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    dataset=dict(
        data_root='./data/avec_2013',
        seg_len=INPUT_SEQUENCE_LEN,
        steps=STEPS,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00001, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[1, 8], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

val_evaluator = [
    dict(type='AvecAccuracy'),
    # dict(type='ClsConfusionMatrix', num_classes=4),
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator