# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9),
    # optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    # clip_grad=dict(max_norm=20, norm_type=2),
)

# learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[60,], gamma=0.1)

# learning policy
warmup_epochs = 20
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
train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
