# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
    )

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[10, 13], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=16, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
