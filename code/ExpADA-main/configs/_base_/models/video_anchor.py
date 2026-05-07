# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="VideoAnchor_1"),
    neck=dict(
        type="VideoAnchorNeck",
        in_channels=1024,
        out_channels=4,
        act_cfg=dict(type="ReLU"),
        teacher=dict(
            type="LinearClsRegHead",
            # pretrained="saved_model/20240416_173204_rmse7.76/0416_epoch_25_rmse7.76_cls0.62.pth",
            in_channels=1024,
            num_classes=4,
            loss=dict(type="RegressionLoss", loss_weight=1.0),
        ),
    ),
    head=dict(
        type="VideoAnchorHead",
        # num_classes=1,
        # in_channels=1024,
        loss=dict(type="RegressionLoss", loss_weight=0.5, rmse=True),
    ),
)
