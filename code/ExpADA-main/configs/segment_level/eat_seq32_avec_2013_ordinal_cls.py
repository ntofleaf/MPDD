_base_ = [
    '../_base_/models/eat.py',
    '../_base_/datasets/avec.py',
    '../_base_/schedules/avec_adam.py',
    '../_base_/default_runtime.py',
]

INPUT_SEQUENCE_LEN, STEPS = 32, 16
BATCH_SIZE, WORKERS = 16, 8

model = dict(
    # pretrained="saved_model/epoch_2_avec_13_10K_rmse_9.0649.pth",
    neck=dict(
        type="EatNeck",
        seq_len=INPUT_SEQUENCE_LEN,
        head=8,
        spatial_h=7,
        spatial_w=7,
    ),
    head=dict(
        type="OrdinalClsRegHead",
        in_channels=1024,
        num_classes=4,
        loss=dict(type="RegressionLoss", loss_weight=1.0),
    ),
)

bgr_mean=[127.5, 127.5, 127.5]
bgr_std=[127.5, 127.5, 127.5]
train_pipeline = [

    dict(type='SequentialRandomFlip', flip_ratio=0.5),
    dict(type='SequentialColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    # dict(
    #     type='SequentialRandomErasing',
    #     erase_prob=0.125,
    #     mode='rand',
    #     min_area_ratio=0.02,
    #     max_area_ratio=1 / 16,
    #     fill_color=bgr_mean,
    #     fill_std=bgr_std
    # ),
    dict(type='PackInputs', algorithm_keys=["subj_id", 'gt_cls_label',]),
]

test_pipeline = [
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
        cls_label_type='ordinal',
        add_resample=False,
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
        cls_label_type='ordinal',
        add_resample=False,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-5, weight_decay=0.0001))

# learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[8, 12], gamma=0.1)

warmup_epochs = 1
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        # T_max=20,
        eta_min=1e-7,
        by_epoch=True,
        begin=warmup_epochs,
        # convert_to_iter_based=True,
    )
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=11, val_interval=1)

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=952292171, deterministic=True)

val_evaluator = [
    dict(type='AvecAccuracy'),
    # dict(type='OrdinalClsConfusionMatrix', num_classes=4),
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator