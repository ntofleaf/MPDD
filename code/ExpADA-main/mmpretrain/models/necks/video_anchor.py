import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class VideoAnchorNeck(BaseModule):
    """Linear neck with Dimension projection.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        gap_dim (int): Dimensions of each sample channel, can be one of
            {0, 1, 2, 3}. Defaults to 0.
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Defaults to dict(type='BN1d').
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Defaults to None.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 gap_dim: int = 0,
                 norm_cfg: Optional[dict] = dict(type='BN1d'),
                 act_cfg: Optional[dict] = None,
                 teacher: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 top_percent_samples: float = 0.2):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.top_p_samples = top_percent_samples
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)
        if teacher is not None:
            self.teacher = MODELS.build(teacher)
            self.teacher = self.teacher.cuda().eval()
        assert gap_dim in [0, 1, 2, 3], 'GlobalAveragePooling dim only ' \
            f'support {0, 1, 2, 3}, get {gap_dim} instead.'
        if gap_dim == 0:
            self.gap = nn.Identity()
        elif gap_dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif gap_dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.cls_fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.cls_fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_channels / 2), out_features=out_channels),
        )
        # self.smp_fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.smp_fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_channels / 2), out_features=out_channels),
        )

        self.reg_proj_branch = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
        )

        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = nn.Identity()

        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = nn.Identity()

    def forward(self, inputs: Union[Tuple,
                                    torch.Tensor]) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Union[Tuple, torch.Tensor]): The features extracted from
                the backbone. Multiple stage inputs are acceptable but only
                the last stage will be used.

        Returns:
            Tuple[torch.Tensor]: A tuple of output features.
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `LinearNeck` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        x = self.gap(inputs)
        # reg_tea, cls_tea, emb = self.teacher(inputs)

        # x = x.view(x.size(0), -1)
        out_cls = self.cls_fc(x)
        out_smp = self.smp_fc(x)
        # loss 1 -------------------------------------------------------------
        # weekly supervised learning
        on_cls_pred = torch.softmax(out_cls, dim=1)
        on_smp_pred = torch.softmax(out_smp, dim=0)
        on_smp_cls = on_cls_pred * on_smp_pred
        smp_on_cls_pred = torch.sum(on_smp_cls, dim=0).clip(0, 1)

        # loss 2 -------------------------------------------------------------
        # strong supervised learning for cls and reg
        #   decide the most confident depression level(cls)
        #   from all samples
        cls_id = torch.argmax(smp_on_cls_pred)
        # cls_id = torch.argmax(on_cls_pred.mean(dim=0))
        # cls_weight = torch.softmax(on_cls_pred[:, cls_id].unsqueeze(1), dim=0)

        # reg_pred_weighted = self.reg_proj_branch(inputs) * cls_weight
        # reg_pred_weighted = reg_pred_weighted.sum(dim=0).unsqueeze(0)

        # -------------------------------------------------------------------
        # select the most confident sample default top_p_samples
        # -------------------------------------------------------------------        
        # select the most confident sample default top 20%
        # conf_, index_ = torch.sort(on_smp_pred[:, cls_id], descending=True)
        # the indexes selected by the following line are the same as the previous line
        conf, index = torch.sort(on_smp_cls[:, cls_id], descending=True)
        top_p = int(len(conf) * self.top_p_samples)
        top_p_index = index[:top_p]
        # top_p_conf = conf[:top_p]

        # select the most confident sample from the output tensor
        # ignort other samples which are not confident or less relevant
        # to the deppression level. In this way, we can suppress the non-
        # relevant samples (noise).
        cls_pred = torch.sigmoid(out_cls)
        # cls_pred = cls_pred[top_p_index]

        # select the most confident sample from the inputs tensor
        reg_pred = self.reg_proj_branch(inputs)
        reg_pred = reg_pred[top_p_index]
        # # TODO: check the mean or sum
        # reg_pred = reg_pred.mean(dim=0).unsqueeze(0)
        # cls_pred, reg_pred = cls_tea, reg_tea
        return smp_on_cls_pred, cls_pred, reg_pred
