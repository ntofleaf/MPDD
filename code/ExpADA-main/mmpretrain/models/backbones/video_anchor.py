import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class VideoAnchor(BaseBackbone):
    """Video anchor backbone.

    """

    def __init__(self, no_anchor=False):
        super(VideoAnchor, self).__init__()
        self.no_anchor = no_anchor
        self.conv_window_1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        self.conv_final = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

    def forward(self, x):
        x1, x2, x3, x5 = x
        x1 = self.conv_window_1(x1)
        if self.no_anchor:
            return (x1.squeeze(), )
        x2 = self.conv_window_2(x2)
        x3 = self.conv_window_3(x3)
        x5 = self.conv_window_5(x5)
        feat = torch.cat([x1, x2, x3, x5], dim=0).squeeze()

        return feat

# 在cat之前每个x+cab
@MODELS.register_module()
class VideoAnchor_1(BaseBackbone):
    """Video anchor backbone.

    """

    def __init__(self, no_anchor=False, reduction=16):
        super(VideoAnchor_1, self).__init__()
        self.no_anchor = no_anchor
        self.conv_window_1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        self.conv_final = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        channel = 1024
        self.cab_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.cab_2 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.cab_3 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.cab_5 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2, x3, x5 = x

        x1_tmp = self.conv_window_1(x1)
        x1 = self.cab_1(x1_tmp.squeeze()) * x1_tmp.squeeze(-1)

        if self.no_anchor:
            return (x1.squeeze(), )
        
        x2_tmp = self.conv_window_2(x2)
        x2 = self.cab_2(x2_tmp.squeeze()) * x2_tmp.squeeze(-1)

        x3_tmp = self.conv_window_3(x3)
        x3 = self.cab_3(x3_tmp.squeeze()) * x3_tmp.squeeze(-1)

        x5_tmp = self.conv_window_5(x5)
        x5 = self.cab_5(x5_tmp.squeeze()) * x5_tmp.squeeze(-1)

        feat = torch.cat([x1, x2, x3, x5], dim=0).squeeze()

        return feat

# 在cat之前每个x+transformer
@MODELS.register_module()
class VideoAnchor_2(BaseBackbone):
    """Video anchor backbone.

    """

    def __init__(self, no_anchor=False, nhead=4, layers=1):
        super(VideoAnchor_2, self).__init__()
        self.no_anchor = no_anchor
        self.conv_window_1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=1024, nhead=nhead)
        self.transformer_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=layers)
        
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=1024, nhead=nhead)
        self.transformer_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=layers)

        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=1024, nhead=nhead)
        self.transformer_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=layers)

        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=1024, nhead=nhead)
        self.transformer_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=layers)
        
    def forward(self, x):
        x1, x2, x3, x5 = x
        x1 = self.conv_window_1(x1)
        x1 = self.transformer_1(x1.transpose(1,2)).transpose(1,2)

        if self.no_anchor:
            return (x1.squeeze(), )
        
        x2 = self.conv_window_2(x2)
        x1 = self.transformer_2(x1.transpose(1,2)).transpose(1,2)

        x3 = self.conv_window_3(x3)
        x1 = self.transformer_3(x1.transpose(1,2)).transpose(1,2)

        x5 = self.conv_window_5(x5)
        x1 = self.transformer_5(x1.transpose(1,2)).transpose(1,2)

        feat = torch.cat([x1, x2, x3, x5], dim=0).squeeze()

        return feat

# 在cat过后的feat上面+的cab
@MODELS.register_module()
class VideoAnchor_3(BaseBackbone):
    """Video anchor backbone.

    """

    def __init__(self, no_anchor=False, reduction=16):
        super(VideoAnchor_3, self).__init__()
        self.no_anchor = no_anchor
        self.conv_window_1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        self.conv_final = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        channel = 1024
        self.cab = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2, x3, x5 = x
        x1 = self.conv_window_1(x1)
        if self.no_anchor:
            return (x1.squeeze(), )
        x2 = self.conv_window_2(x2)
        x3 = self.conv_window_3(x3)
        x5 = self.conv_window_5(x5)
        feat = torch.cat([x1, x2, x3, x5], dim=0).squeeze()

        feat = self.cab(feat) * feat

        return feat

@MODELS.register_module()
class VideoAnchor_4(BaseBackbone):
    """Video anchor backbone.

    """

    def __init__(self, no_anchor=False, nhead=4, layers=1):
        super(VideoAnchor_4, self).__init__()
        self.no_anchor = no_anchor
        self.conv_window_1 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.conv_window_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, 5),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        x1, x2, x3, x5 = x
        x1 = self.conv_window_1(x1)
        if self.no_anchor:
            return (x1.squeeze(), )
        x2 = self.conv_window_2(x2)
        x3 = self.conv_window_3(x3)
        x5 = self.conv_window_5(x5)
        feat = torch.cat([x1, x2, x3, x5], dim=0)

        feat = self.transformer(feat.transpose(1,2)).squeeze()

        return feat
