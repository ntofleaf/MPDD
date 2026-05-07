_base_ = [
    '../_base_/models/eat.py',
    '../_base_/datasets/avec.py',
    '../_base_/schedules/avec_adam.py',
    '../_base_/default_runtime.py',
]

INPUT_SEQUENCE_LEN, STEPS = 64, 32
BATCH_SIZE, WORKERS = 8, 8

model = dict(
    # pretrained="saved_model/epoch_2_avec_13_10K_rmse_9.0649.pth",
    neck=dict(
        type="VatNeck",
        seq_len=INPUT_SEQUENCE_LEN,
        head=8,
        spatial_h=7,
        spatial_w=7,
    ),
)

# data loader and dataset settings
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    dataset=dict(
        data_root='./data/avec_2013',
        seg_len=INPUT_SEQUENCE_LEN,
        steps=STEPS,
    )
)
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    dataset=dict(
        data_root='./data/avec_2013',
        seg_len=INPUT_SEQUENCE_LEN,
        steps=STEPS,
    )
)
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00001, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[5, 8], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=110951288, deterministic=True)