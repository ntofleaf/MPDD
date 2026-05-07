# model settings
model = dict(
    type="ConvTransModel",
    backbone=dict(type="VatBackbone", init_weights=False),
    neck=dict(
        type="VatNeck",
        seq_len=64,
        head=16,
        spatial_h=4,
        spatial_w=4,
    ),
    head=dict(
        type="LinearRegHead",
        num_classes=1,
        in_channels=1024,
        loss=dict(type="RegressionLoss", loss_weight=1.0),
    ),
)
