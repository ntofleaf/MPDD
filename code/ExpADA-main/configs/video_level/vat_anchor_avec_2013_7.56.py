_base_ = [
    '../_base_/datasets/avec_video_anchor.py',
    '../_base_/models/video_anchor.py',
    '../_base_/schedules/video_anchor.py',
    '../_base_/default_runtime.py',
]

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='ResizeEdge', scale=224, edge='short'),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackFeatInputs', algorithm_keys=["subj_id",]),
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='ResizeEdge', scale=224, edge='short'),
    dict(type='PackFeatInputs', algorithm_keys=["subj_id",]),
]

data_root = 'data/avec_2013/extracted_features'
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, data_root=data_root))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, data_root=data_root))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, data_root=data_root))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.8),
    # optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    # clip_grad=dict(max_norm=20, norm_type=2),
)

# learning policy
warmup_epochs = 5
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=924526860, deterministic=True)
